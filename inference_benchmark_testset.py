#!/usr/bin/env python
# coding=utf-8

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_kontext_multiple_images import FluxKontextPipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.loading_utils import load_image
from peft import LoraConfig, set_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from system_config import cfg_path, fitting_lora_checkpoint, flux_model_id, longclip_model_id, texture_lora_checkpoint


DRESSCODE_CATEGORY_BY_ID = {
    "0": "upper_body",
    "1": "lower_body",
    "2": "dresses",
    0: "upper_body",
    1: "lower_body",
    2: "dresses",
    "upper_body": "upper_body",
    "lower_body": "lower_body",
    "dresses": "dresses",
}

DRESSCODE_CATEGORY_CHOICES = ("upper_body", "lower_body", "dresses")

BODY_PROMPT_BY_DATASET = {
    "dresscode": "The person has a slim and tall body shape. ",
    "viton": "The person has a slim and medium-tall body shape. ",
}

DEFAULT_FITTING_LORA = fitting_lora_checkpoint()
DEFAULT_TEXTURE_LORA = texture_lora_checkpoint()


def resolve_lora_dir(path: str) -> tuple[str, str]:
    path_obj = Path(path)
    if path_obj.is_file():
        return str(path_obj.parent), path_obj.name
    return str(path_obj), "pytorch_lora_weights.safetensors"


@dataclass
class BenchmarkItem:
    person_filename: str
    ref_filename: str
    cloth_prompt: str
    dresscode_category: str | None = None


def resolve_dresscode_image_path(dataset_root: str, category_subdir: str, filename: str) -> str:
    candidate_1 = os.path.join(dataset_root, category_subdir, filename)
    if os.path.exists(candidate_1):
        return candidate_1

    candidate_2 = os.path.join(dataset_root, category_subdir, "images", filename)
    if os.path.exists(candidate_2):
        return candidate_2

    return candidate_1


def resolve_viton_path(dataset_root: str, split: str, subdir: str, filename: str) -> str:
    candidate_1 = os.path.join(dataset_root, split, subdir, filename)
    if os.path.exists(candidate_1):
        return candidate_1

    candidate_2 = os.path.join(dataset_root, subdir, filename)
    if os.path.exists(candidate_2):
        return candidate_2

    return candidate_1


def read_dresscode_pairs(pairs_file: str, default_category_id: str | None = None):
    items: list[tuple[str, str, str]] = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue

            parts = raw.split()
            if len(parts) not in (2, 3):
                raise ValueError(
                    f"Bad line format at {pairs_file}:{line_idx}: expected 2 or 3 columns, got {len(parts)}: {raw!r}"
                )

            person_filename, ref_filename = parts[0], parts[1]
            if len(parts) == 3:
                category_id = parts[2]
            else:
                if default_category_id is None:
                    raise ValueError(
                        f"pairs_file has 2 columns but no default category was provided: {pairs_file}:{line_idx}"
                    )
                category_id = default_category_id

            if category_id not in DRESSCODE_CATEGORY_BY_ID:
                raise ValueError(f"Unknown DressCode category id at {pairs_file}:{line_idx}: {category_id!r}")

            items.append((person_filename, ref_filename, DRESSCODE_CATEGORY_BY_ID[category_id]))
    return items


def read_viton_pairs(pairs_file: str):
    items: list[tuple[str, str]] = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue

            parts = raw.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Bad line format at {pairs_file}:{line_idx}: expected 2 columns, got {len(parts)}: {raw!r}"
                )
            items.append((parts[0], parts[1]))
    return items


def read_length_prompts_line_aligned(length_file: str) -> list[str]:
    prompts: list[str] = []
    with open(length_file, "r", encoding="utf-8", errors="ignore") as f:
        for line_idx, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue

            parts = raw.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"Bad line format at {length_file}:{line_idx}: expected >=2 TSV columns, got {len(parts)}: {raw!r}"
                )
            prompts.append("\t".join(parts[1:]).strip())
    return prompts


def read_length_prompts_keyed(length_file: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    with open(length_file, "r", encoding="utf-8", errors="ignore") as f:
        for line_idx, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue

            parts = raw.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"Bad line format at {length_file}:{line_idx}: expected >=2 TSV columns, got {len(parts)}: {raw!r}"
                )
            entries.append((parts[0].strip(), "\t".join(parts[1:]).strip()))
    return entries


def build_prompt(body_prompt: str, cloth_prompt: str) -> str:
    body = (body_prompt or "").strip()
    cloth = (cloth_prompt or "").strip()
    if not body:
        return cloth
    if not cloth:
        return body
    return body + " " + cloth


def build_dresscode_items(args) -> list[BenchmarkItem]:
    if args.cloth_prompt_file is None:
        args.cloth_prompt_file = os.path.join(
            args.dataset_root, args.category, f"dresscode_{args.category}_length.txt"
        )

    pairs = read_dresscode_pairs(args.pairs_file, default_category_id=args.category)
    prompts = read_length_prompts_line_aligned(args.cloth_prompt_file)
    if len(pairs) != len(prompts):
        raise ValueError(
            f"pairs_file and cloth_prompt_file must have the same number of usable lines. "
            f"Got pairs={len(pairs)} prompts={len(prompts)}. "
            f"pairs_file={args.pairs_file} cloth_prompt_file={args.cloth_prompt_file}"
        )

    items: list[BenchmarkItem] = []
    for (person_filename, ref_filename, category_subdir), cloth_prompt in zip(pairs, prompts):
        if category_subdir != args.category:
            continue
        items.append(
            BenchmarkItem(
                person_filename=person_filename,
                ref_filename=ref_filename,
                cloth_prompt=cloth_prompt,
                dresscode_category=category_subdir,
            )
        )

    if not items:
        raise RuntimeError(
            f"No DressCode items found for --category={args.category}. Check category ids in {args.pairs_file}."
        )
    return items


def build_viton_items(args) -> list[BenchmarkItem]:
    if args.pairs_file is None:
        candidate = os.path.join(args.dataset_root, args.split, f"{args.split}_unpairs.txt")
        if os.path.exists(candidate):
            args.pairs_file = candidate
        else:
            args.pairs_file = os.path.join(args.dataset_root, args.split, "test_unpairs.txt")

    if args.cloth_prompt_file is None:
        args.cloth_prompt_file = os.path.join(args.dataset_root, args.split, "viton_upper_body_length.txt")

    pairs = read_viton_pairs(args.pairs_file)
    length_entries = read_length_prompts_keyed(args.cloth_prompt_file)

    aligned = (
        len(pairs) == len(length_entries)
        and all(pair[1] == length_entries[i][0] for i, pair in enumerate(pairs))
    )

    prompt_by_cloth: dict[str, str] = {}
    if not aligned:
        for key, prompt in length_entries:
            prompt_by_cloth.setdefault(key, prompt)

    items: list[BenchmarkItem] = []
    missing_prompt_count = 0
    for idx, (person_filename, cloth_filename) in enumerate(pairs):
        if aligned:
            cloth_prompt = length_entries[idx][1]
        else:
            cloth_prompt = prompt_by_cloth.get(cloth_filename, "")
            if not cloth_prompt:
                missing_prompt_count += 1
        items.append(
            BenchmarkItem(
                person_filename=person_filename,
                ref_filename=cloth_filename,
                cloth_prompt=cloth_prompt,
            )
        )

    if missing_prompt_count:
        print(f"[warn] missing cloth prompts for {missing_prompt_count}/{len(items)} items")

    if not items:
        raise RuntimeError(f"No VITON items loaded from pairs_file: {args.pairs_file}")
    return items


def torch_dtype_from_arg(dtype: str):
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def get_process_shard_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return local_rank, rank, world_size


def load_pipeline(args, device: str, dtype: torch.dtype) -> FluxKontextPipeline:
    flux_kontext_model_id = flux_model_id()
    longclip_repo_id = longclip_model_id()
    revision = "main"

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        flux_kontext_model_id, subfolder="scheduler", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        longclip_repo_id, torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(longclip_repo_id)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        flux_kontext_model_id, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision
    )
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        flux_kontext_model_id, subfolder="tokenizer_2", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(
        flux_kontext_model_id, subfolder="vae", torch_dtype=dtype, revision=revision
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        flux_kontext_model_id, subfolder="transformer", torch_dtype=dtype, revision=revision
    )

    fitting_lora_dir, fitting_weight_name = resolve_lora_dir(args.fitting_lora_dir)
    texture_lora_dir, texture_weight_name = resolve_lora_dir(args.texture_lora_dir)
    te2_lora_path = os.path.join(fitting_lora_dir, "text_encoder_2_lora.pth")
    if os.path.exists(te2_lora_path):
        text_lora_config_t5 = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            init_lora_weights="gaussian",
            target_modules=["q", "k", "v", "o"],
        )
        text_encoder_2.add_adapter(text_lora_config_t5)
        te2_state_dict = torch.load(te2_lora_path, map_location="cpu")
        set_peft_model_state_dict(text_encoder_2, te2_state_dict, adapter_name="default")
        if hasattr(text_encoder_2, "set_adapter"):
            text_encoder_2.set_adapter("default")
    else:
        print(f"[warn] text_encoder_2_lora.pth not found at: {te2_lora_path}")

    text_encoder.to(device, dtype=dtype)
    text_encoder_2.to(device, dtype=dtype)
    vae.to(device, dtype=dtype)
    transformer.to(device, dtype=dtype)

    pipe = FluxKontextPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=transformer,
    )

    pipe.load_lora_weights(
        fitting_lora_dir, weight_name=fitting_weight_name, adapter_name="default"
    )
    pipe.load_lora_weights(
        texture_lora_dir, weight_name=texture_weight_name, adapter_name="texture"
    )
    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters(
            ["default", "texture"],
            adapter_weights=[args.fitting_lora_weight, args.texture_lora_weight],
        )

    pipe.tokenizer_max_length = 248
    pipe.to(device)
    return pipe


def resolve_item_paths(args, item: BenchmarkItem) -> tuple[str, str]:
    if args.dataset == "dresscode":
        category = item.dresscode_category or args.category
        person_path = resolve_dresscode_image_path(args.dataset_root, category, item.person_filename)
        ref_path = resolve_dresscode_image_path(args.dataset_root, category, item.ref_filename)
        return person_path, ref_path

    person_path = resolve_viton_path(args.dataset_root, args.split, "image", item.person_filename)
    ref_path = resolve_viton_path(args.dataset_root, args.split, "cloth", item.ref_filename)
    return person_path, ref_path


def resolve_output_path(args, item: BenchmarkItem) -> str:
    person_base = os.path.splitext(os.path.basename(item.person_filename))[0]
    ref_base = os.path.splitext(os.path.basename(item.ref_filename))[0]

    if args.dataset == "dresscode":
        out_dir = os.path.join(args.output_dir, args.category, "result_fitvton")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{person_base}_{ref_base}.jpg")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_ext = (args.output_ext or "png").lower().lstrip(".")
    return os.path.join(out_dir, f"{person_base}_{ref_base}.{out_ext}")


def image_save_kwargs(args, out_path: str) -> dict:
    if out_path.lower().endswith(".png"):
        cl = max(0, min(9, int(args.png_compress_level)))
        return {"compress_level": cl, "optimize": bool(args.png_optimize)}
    if out_path.lower().endswith((".jpg", ".jpeg")):
        q = max(1, min(95, int(args.jpeg_quality)))
        return {"quality": q, "subsampling": int(args.jpeg_subsampling), "optimize": True}
    return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DressCode or VITON benchmark inference with FluxKontextPipeline."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["dresscode", "viton"],
        help="Benchmark dataset to run.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Dataset root folder. Defaults depend on --dataset.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="upper_body",
        choices=list(DRESSCODE_CATEGORY_CHOICES),
        help="[dresscode] Category folder to run.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="[viton] Dataset split folder under dataset_root.",
    )
    parser.add_argument(
        "--pairs_file",
        type=str,
        default=None,
        help="Pairs list file. Format depends on --dataset.",
    )
    parser.add_argument(
        "--cloth_prompt_file",
        "--length_file",
        dest="cloth_prompt_file",
        type=str,
        default=None,
        help="TSV cloth prompt fragments. Format depends on --dataset.",
    )
    parser.add_argument(
        "--fitting_lora_dir",
        type=str,
        default=DEFAULT_FITTING_LORA,
        help="Fitting LoRA checkpoint file or directory.",
    )
    parser.add_argument(
        "--texture_lora_dir",
        type=str,
        default=DEFAULT_TEXTURE_LORA,
        help="Texture LoRA checkpoint file or directory.",
    )
    parser.add_argument("--fitting_lora_weight", type=float, default=0.8)
    parser.add_argument("--texture_lora_weight", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default=None, help="Output root directory.")
    parser.add_argument(
        "--output_ext",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="[viton] Output image extension.",
    )
    parser.add_argument("--png_compress_level", type=int, default=9)
    parser.add_argument("--png_optimize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--jpeg_subsampling", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def apply_dataset_defaults(args):
    if args.dataset_root is None:
        args.dataset_root = (
            cfg_path("datasets", "dresscode_root")
            if args.dataset == "dresscode"
            else cfg_path("datasets", "viton_root")
        )
    if args.output_dir is None:
        args.output_dir = args.dataset_root
    if args.dataset == "dresscode" and args.pairs_file is None:
        args.pairs_file = os.path.join(args.dataset_root, "second_training_data", "train_pairs_unpaired.txt")


def main(args):
    apply_dataset_defaults(args)

    dtype = torch_dtype_from_arg(args.dtype)
    local_rank, rank, world_size = get_process_shard_info()

    if torch.cuda.is_available() and world_size > 1:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    if args.dataset == "dresscode":
        items = build_dresscode_items(args)
    else:
        items = build_viton_items(args)

    if args.start_index < 0 or args.start_index >= len(items):
        raise ValueError(f"--start_index out of range: {args.start_index} (num items: {len(items)})")

    end = min(len(items), args.start_index + max(0, args.max_samples))
    if world_size > 1:
        print(
            f"[dist] world_size={world_size} rank={rank} local_rank={local_rank} device={device} "
            f"processing indices in [{args.start_index}, {end})"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    pipe = load_pipeline(args, device=device, dtype=dtype)
    body_prompt = BODY_PROMPT_BY_DATASET[args.dataset]

    for global_idx in range(args.start_index, end):
        if world_size > 1 and (global_idx % world_size) != rank:
            continue

        item = items[global_idx]
        out_path = resolve_output_path(args, item)
        if args.skip_existing and os.path.exists(out_path):
            print(f"[skip] exists: {out_path}")
            continue

        person_path, ref_path = resolve_item_paths(args, item)
        if not os.path.exists(person_path):
            print(f"[skip] missing person image: {person_path}")
            continue
        if not os.path.exists(ref_path):
            print(f"[skip] missing reference image: {ref_path}")
            continue

        prompt = build_prompt(body_prompt, item.cloth_prompt)
        input_image = load_image(person_path)
        ref_image = load_image(ref_path)
        sample_gen = torch.Generator(device=device).manual_seed(args.seed + global_idx)

        with torch.inference_mode():
            image = pipe(
                multiple_images=[(input_image, ref_image)],
                prompt=prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=sample_gen,
                num_images_per_prompt=1,
                _auto_resize=False,
            ).images[0]

        image.save(out_path, **image_save_kwargs(args, out_path))
        print(f"[ok] {out_path}")


if __name__ == "__main__":
    main(parse_args())
