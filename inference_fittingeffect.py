import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers.loaders.lora_base import disable_lora_for_text_encoder
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_kontext_multiple_images import FluxKontextPipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.loading_utils import load_image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from system_config import (
    cfg_path,
    fitting_lora_checkpoint,
    fittingeffect_subpath,
    flux_model_id,
    local_model_override,
    longclip_model_id,
    texture_lora_checkpoint,
)


DEFAULT_FITTING_LORA = fitting_lora_checkpoint()
DEFAULT_TEXTURE_LORA = texture_lora_checkpoint()
DEFAULT_FEMALE_ROOT = fittingeffect_subpath("female")
DEFAULT_MALE_ROOT = fittingeffect_subpath("male")


BODY_LABEL_BY_GENDER_ID: dict[str, dict[str, str]] = {
    "female": {
        "0": "slim medium-tall",
        "1": "slim tall",
        "2": "average medium-short",
        "3": "heavy tall",
        "4": "slim short",
    },
    "male": {
        "0": "plus-size medium-short",
        "1": "average tall",
        "2": "slim tall",
        "3": "plus-size medium-short",
        "4": "average medium-tall",
    },
}


GARMENT_INFO_BY_GENDER_CLOTH_ID: dict[str, dict[str, tuple[str, str]]] = {
    "female": {
        "upper_0": ("short-length", "upper"),
        "upper_1": ("short-length", "upper"),
        "upper_2": ("long-length", "upper"),
        "lower_0": ("short-length", "lower"),
        "lower_1": ("long-length", "lower"),
        "lower_2": ("long-length", "lower"),
        "wholebody_0": ("long-length", "dress"),
        "wholebody_1": ("short-length", "dress"),
        "dress_0": ("long-length", "dress"),
        "dress_1": ("short-length", "dress"),
    },
    "male": {
        "upper_0": ("long-length", "upper"),
        "upper_1": ("long-length", "upper"),
        "upper_2": ("short-length", "upper"),
        "upper_3": ("short-length", "upper"),
        "lower_0": ("long-length", "lower"),
        "lower_1": ("long-length", "lower"),
        "lower_2": ("short-length", "lower"),
        "lower_3": ("short-length", "lower"),
    },
}

STYLE_WORDS = {
    "tucked_in": "tucked in",
    "untucked": "untucked",
    "one_piece": "one-piece",
}


NAME_RE = re.compile(r"^(female|male)_(\d+)_(.+)_pose_(\d+)$")


@dataclass
class TripletItem:
    gender: str
    csv_path: str
    row_index: int
    source_person: str
    cloth: str
    target_person: str


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


def should_process_index(global_idx: int, args, rank: int, world_size: int) -> bool:
    if args.num_shards > 1:
        return global_idx % args.num_shards == args.shard_index
    if world_size > 1:
        return global_idx % world_size == rank
    return True


def manifest_name_for_run(args, rank: int, world_size: int) -> str:
    if args.num_shards > 1:
        return f"manifest_shard{args.shard_index:02d}.jsonl"
    if world_size > 1:
        return f"manifest_rank{rank}.jsonl"
    return "manifest.jsonl"


def parse_person_id(person_name: str) -> tuple[str, str]:
    match = NAME_RE.match(person_name)
    if not match:
        raise ValueError(f"Bad person id: {person_name!r}")
    return match.group(1), match.group(2)


def body_prompt_from_person_id(person_name: str) -> str:
    gender, person_idx = parse_person_id(person_name)
    labels = BODY_LABEL_BY_GENDER_ID.get(gender, {})
    if person_idx not in labels:
        raise KeyError(f"No body prompt mapping for {gender}_{person_idx}")
    label = labels[person_idx]
    shape, height = label.split(" ", 1)
    return f"The person is a {height} and {shape} {gender}."


def garment_info_from_cloth_id(gender: str, cloth_id: str) -> tuple[str, str]:
    garments = GARMENT_INFO_BY_GENDER_CLOTH_ID.get(gender, {})
    if cloth_id not in garments:
        raise KeyError(f"No cloth prompt mapping for {gender}/{cloth_id}")
    return garments[cloth_id]


def style_from_garment_type(garment_type: str) -> str:
    return "one_piece" if garment_type == "dress" else "untucked"


def build_prompt(source_person: str, cloth_id: str) -> str:
    gender, _ = parse_person_id(source_person)
    body_prompt = body_prompt_from_person_id(source_person).strip()
    length, garment_type = garment_info_from_cloth_id(gender, cloth_id)
    style = style_from_garment_type(garment_type)
    cloth_prompt = f"The cloth is a {length} {garment_type} garment."
    wear_prompt = f"The wearing style is {STYLE_WORDS[style]}."
    return f"{body_prompt} {cloth_prompt} {wear_prompt}"


def read_triplets(csv_path: str, expected_gender: str | None = None) -> list[TripletItem]:
    items: list[TripletItem] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"source_person", "cloth", "target_person"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{csv_path} must contain columns: {sorted(required)}")

        for row_index, row in enumerate(reader, start=1):
            source_person = row["source_person"].strip()
            cloth = row["cloth"].strip()
            target_person = row["target_person"].strip()
            gender, _ = parse_person_id(source_person)
            target_gender, _ = parse_person_id(target_person)
            if gender != target_gender:
                raise ValueError(
                    f"Gender mismatch at {csv_path}:{row_index + 1}: "
                    f"source={source_person!r}, target={target_person!r}"
                )
            if expected_gender is not None and gender != expected_gender:
                raise ValueError(
                    f"Unexpected gender at {csv_path}:{row_index + 1}: "
                    f"source={source_person!r}, target={target_person!r}, expected={expected_gender!r}"
                )
            build_prompt(source_person, cloth)
            items.append(
                TripletItem(
                    gender=gender,
                    csv_path=csv_path,
                    row_index=row_index,
                    source_person=source_person,
                    cloth=cloth,
                    target_person=target_person,
                )
            )
    return items


def resolve_image_path(dataset_root: str, subdir: str, stem: str, ext: str) -> str:
    return str(Path(dataset_root) / subdir / f"{stem}.{ext.lstrip('.')}")


def output_name(item: TripletItem, output_ext: str) -> str:
    return f"{item.source_person}_{item.cloth}.{output_ext.lstrip('.')}"


def resolve_lora_dir(path: str) -> tuple[Path, str]:
    path_obj = Path(path)
    if path_obj.is_file():
        return path_obj.parent, path_obj.name
    return path_obj, "pytorch_lora_weights.safetensors"


def resolve_hf_snapshot(repo_id: str, revision: str = "main") -> Path | None:
    """Resolve a local model path from system.json or HuggingFace cache when available."""
    configured = local_model_override(repo_id)
    if configured is not None:
        return configured

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    cache_dir = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"
    refs_file = cache_dir / "refs" / revision
    if refs_file.exists():
        snapshot = cache_dir / "snapshots" / refs_file.read_text().strip()
        if snapshot.exists():
            return snapshot

    snapshots_dir = cache_dir / "snapshots"
    if snapshots_dir.exists():
        candidates = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    return None


def load_pipeline(args, device: str, dtype):
    flux_kontext_model_id = flux_model_id()
    longclip_repo_id = longclip_model_id()
    revision = "main"

    local_flux_snapshot = resolve_hf_snapshot(flux_kontext_model_id, revision=revision)
    local_longclip = resolve_hf_snapshot(longclip_repo_id, revision=revision)
    flux_model_path = str(local_flux_snapshot) if local_flux_snapshot is not None else flux_kontext_model_id
    longclip_repo = str(local_longclip) if local_longclip is not None else longclip_repo_id
    local_kwargs = {"local_files_only": True} if local_flux_snapshot is not None else {}
    longclip_local_kwargs = {"local_files_only": True} if local_longclip is not None else {}

    if local_flux_snapshot is not None:
        print(f"Loading FLUX base from local cache: {local_flux_snapshot}")
    if local_longclip is not None:
        print(f"Loading LongCLIP from local cache: {local_longclip}")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        flux_model_path, subfolder="scheduler", revision=revision, **local_kwargs
    )
    text_encoder = CLIPTextModel.from_pretrained(
        longclip_repo,
        torch_dtype=dtype,
        **longclip_local_kwargs,
    )
    tokenizer = CLIPTokenizer.from_pretrained(longclip_repo, **longclip_local_kwargs)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        flux_model_path,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
        revision=revision,
        **local_kwargs,
    )
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        flux_model_path, subfolder="tokenizer_2", revision=revision, **local_kwargs
    )
    vae = AutoencoderKL.from_pretrained(
        flux_model_path, subfolder="vae", torch_dtype=dtype, revision=revision, **local_kwargs
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        flux_model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        revision=revision,
        **local_kwargs,
    )

    pipe = FluxKontextPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=transformer,
    )

    lora_dir, weight_name = resolve_lora_dir(args.fitting_lora_dir)
    fitting_lora_path = lora_dir / weight_name
    if not fitting_lora_path.exists():
        raise FileNotFoundError(f"Fitting LoRA weights not found: {fitting_lora_path}")
    pipe.load_lora_weights(str(lora_dir), weight_name=weight_name, adapter_name="default")
    disable_lora_for_text_encoder(pipe.text_encoder)
    print(f"Loaded transformer-only fitting LoRA: {fitting_lora_path}")

    use_texture = not args.no_texture_lora and bool(args.texture_lora_dir)
    if use_texture:
        texture_dir, texture_weight_name = resolve_lora_dir(args.texture_lora_dir)
        texture_lora_path = texture_dir / texture_weight_name
        if not texture_lora_path.exists():
            raise FileNotFoundError(f"Texture LoRA weights not found: {texture_lora_path}")
        pipe.load_lora_weights(
            str(texture_dir),
            weight_name=texture_weight_name,
            adapter_name="texture",
        )
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(
                ["default", "texture"],
                adapter_weights=[args.fitting_lora_weight, args.texture_lora_weight],
            )
        print(
            f"Loaded texture LoRA: {texture_lora_path} "
            f"(weights={args.fitting_lora_weight},{args.texture_lora_weight})"
        )
    elif hasattr(pipe, "set_adapters"):
        pipe.set_adapters("default")

    pipe.tokenizer_max_length = 248
    pipe.to(device, dtype=dtype)
    return pipe


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch FLUX-Kontext inference on FittingEffect-style triplet CSVs with long_vton prompts, "
            "transformer-only fitting LoRA, and optional texture LoRA."
        )
    )
    parser.add_argument("--female_root", type=str, default=DEFAULT_FEMALE_ROOT)
    parser.add_argument("--male_root", type=str, default=DEFAULT_MALE_ROOT)
    parser.add_argument(
        "--female_csv",
        type=str,
        default=None,
        help="CSV with source_person,cloth,target_person. Required when --gender is all or female.",
    )
    parser.add_argument(
        "--male_csv",
        type=str,
        default=None,
        help="CSV with source_person,cloth,target_person. Required when --gender is all or male.",
    )
    parser.add_argument(
        "--triples_csv",
        type=str,
        default=None,
        help="Optional mixed female/male CSV. When set, overrides --female_csv/--male_csv.",
    )
    parser.add_argument("--gender", choices=["all", "female", "male"], default="all")
    parser.add_argument(
        "--flat_output",
        action="store_true",
        help="Write all outputs directly under --output_dir (no female/male subfolders).",
    )
    parser.add_argument("--human_subdir", type=str, default="human")
    parser.add_argument("--cloth_subdir", type=str, default="cloth")
    parser.add_argument("--human_ext", type=str, default="jpg")
    parser.add_argument("--cloth_ext", type=str, default="jpg")
    parser.add_argument("--output_dir", type=str, default=cfg_path("outputs", "fittingeffect"))

    parser.add_argument("--fitting_lora_dir", type=str, default=DEFAULT_FITTING_LORA)
    parser.add_argument("--texture_lora_dir", type=str, default=DEFAULT_TEXTURE_LORA)
    parser.add_argument("--lora_weight_name", type=str, default="pytorch_lora_weights.safetensors")
    parser.add_argument("--texture_lora_weight_name", type=str, default="pytorch_lora_weights.safetensors")
    parser.add_argument(
        "--fitting_lora_weight",
        "--default_lora_weight",
        dest="fitting_lora_weight",
        type=float,
        default=0.8,
    )
    parser.add_argument("--texture_lora_weight", type=float, default=0.2)
    parser.add_argument("--no_texture_lora", action="store_true", help="Load only fitting LoRA.")

    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output_ext", type=str, default="png", choices=["png", "jpg", "jpeg"])
    parser.add_argument("--png_compress_level", type=int, default=9)
    parser.add_argument("--png_optimize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--jpeg_subsampling", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--dry_run", action="store_true", help="Validate triplets/paths/prompts without loading the model.")
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Manual multi-GPU sharding: total number of parallel workers.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Manual multi-GPU sharding: worker index in [0, num_shards).",
    )
    return parser.parse_args()


def collect_items(args) -> list[TripletItem]:
    if args.triples_csv:
        items = read_triplets(args.triples_csv)
        if args.gender == "female":
            items = [item for item in items if item.gender == "female"]
        elif args.gender == "male":
            items = [item for item in items if item.gender == "male"]
        return items

    items: list[TripletItem] = []
    if args.gender in {"all", "female"}:
        if not args.female_csv:
            raise ValueError("--female_csv is required when --gender is all or female.")
        items.extend(read_triplets(args.female_csv, "female"))
    if args.gender in {"all", "male"}:
        if not args.male_csv:
            raise ValueError("--male_csv is required when --gender is all or male.")
        items.extend(read_triplets(args.male_csv, "male"))
    return items


def root_for_gender(args, gender: str) -> str:
    return args.female_root if gender == "female" else args.male_root


def save_image(image, out_path: str, args):
    save_kwargs = {}
    if out_path.lower().endswith(".png"):
        compress_level = max(0, min(9, int(args.png_compress_level)))
        save_kwargs.update({"compress_level": compress_level, "optimize": bool(args.png_optimize)})
    elif out_path.lower().endswith((".jpg", ".jpeg")):
        quality = max(1, min(95, int(args.jpeg_quality)))
        save_kwargs.update({"quality": quality, "subsampling": int(args.jpeg_subsampling), "optimize": True})
    image.save(out_path, **save_kwargs)


def main(args):
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must be in [0, num_shards)")

    local_rank, rank, world_size = get_process_shard_info()
    if torch.cuda.is_available() and world_size > 1:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    items = collect_items(args)
    if not items:
        raise RuntimeError("No triplets loaded.")
    if args.start_index < 0 or args.start_index >= len(items):
        raise ValueError(f"--start_index out of range: {args.start_index} (num items: {len(items)})")
    end = len(items) if args.max_samples is None else min(len(items), args.start_index + max(0, args.max_samples))
    items = items[args.start_index:end]

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / manifest_name_for_run(args, rank, world_size)

    is_leader = args.shard_index == 0 and rank == 0
    if is_leader:
        print(f"Loaded {len(items)} triplet(s) from index range [{args.start_index}, {end}).")
        print(f"Output root: {output_root}")
    if args.num_shards > 1:
        print(
            f"[shard] {args.shard_index}/{args.num_shards} device={device} "
            f"manifest={manifest_path.name}"
        )
    elif world_size > 1:
        print(f"[dist] world_size={world_size} rank={rank} local_rank={local_rank} device={device}")

    missing = []
    preview = []
    for offset, item in enumerate(items):
        root = root_for_gender(args, item.gender)
        source_path = resolve_image_path(root, args.human_subdir, item.source_person, args.human_ext)
        cloth_path = resolve_image_path(root, args.cloth_subdir, item.cloth, args.cloth_ext)
        target_path = resolve_image_path(root, args.human_subdir, item.target_person, args.human_ext)
        prompt = build_prompt(item.source_person, item.cloth)
        if offset < 8:
            preview.append((item.gender, item.source_person, item.cloth, prompt))
        for label, path in [("source", source_path), ("cloth", cloth_path), ("target", target_path)]:
            if not os.path.exists(path):
                missing.append((label, path))

    if is_leader:
        print("Prompt preview:")
        for gender, source_person, cloth, prompt in preview:
            print(f"  [{gender}] {source_person} + {cloth}: {prompt}")

    if missing:
        for label, path in missing[:20]:
            print(f"[missing] {label}: {path}")
        raise FileNotFoundError(f"Found {len(missing)} missing path(s).")

    if args.dry_run:
        assigned = sum(
            1
            for local_idx in range(args.start_index, end)
            if should_process_index(local_idx, args, rank, world_size)
        )
        print(
            f"Dry run complete for shard {args.shard_index}/{args.num_shards}; "
            f"assigned_samples={assigned}; model was not loaded."
        )
        return

    dtype = torch_dtype_from_arg(args.dtype)
    pipe = load_pipeline(args, device=device, dtype=dtype)

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for local_idx, item in enumerate(items, start=args.start_index):
            if not should_process_index(local_idx, args, rank, world_size):
                continue

            root = root_for_gender(args, item.gender)
            source_path = resolve_image_path(root, args.human_subdir, item.source_person, args.human_ext)
            cloth_path = resolve_image_path(root, args.cloth_subdir, item.cloth, args.cloth_ext)
            target_path = resolve_image_path(root, args.human_subdir, item.target_person, args.human_ext)
            prompt = build_prompt(item.source_person, item.cloth)

            out_dir = output_root if args.flat_output else output_root / item.gender
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / output_name(item, args.output_ext)
            if args.skip_existing and out_path.exists():
                print(f"[skip] exists: {out_path}")
                continue

            input_image = load_image(source_path)
            ref_image = load_image(cloth_path)
            sample_gen = torch.Generator(device=device).manual_seed(args.seed + local_idx)

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

            save_image(image, str(out_path), args)
            row = {
                "output": str(out_path),
                "gender": item.gender,
                "csv_path": item.csv_path,
                "row_index": item.row_index,
                "source_person": item.source_person,
                "cloth": item.cloth,
                "target_person": item.target_person,
                "source_path": source_path,
                "cloth_path": cloth_path,
                "target_path": target_path,
                "prompt": prompt,
                "fitting_lora_dir": args.fitting_lora_dir,
                "texture_lora_dir": None if args.no_texture_lora else args.texture_lora_dir,
                "fitting_lora_weight": args.fitting_lora_weight,
                "texture_lora_weight": args.texture_lora_weight,
                "seed": args.seed + local_idx,
            }
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest.flush()
            print(f"[ok] {out_path}")

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main(parse_args())
