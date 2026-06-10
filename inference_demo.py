#!/usr/bin/env python
# coding=utf-8
"""Single-sample VTON inference with long_vton prompts and dual transformer LoRA."""

import argparse
import json
import re
from itertools import product
from pathlib import Path

import torch
from diffusers.loaders.lora_base import disable_lora_for_text_encoder
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_kontext_multiple_images import FluxKontextPipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.loading_utils import load_image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from system_config import cfg_path, fitting_lora_checkpoint, flux_model_id, longclip_model_id, texture_lora_checkpoint


DEFAULT_FITTING_LORA = fitting_lora_checkpoint()
DEFAULT_TEXTURE_LORA = texture_lora_checkpoint()

GENDER_CHOICES = ("female", "male")
SHAPE_CHOICES = ("slim", "average", "heavy", "plus-size")
HEIGHT_CHOICES = ("short", "medium-short", "medium-tall", "tall")
LENGTH_CHOICES = ("short-length", "long-length")
GARMENT_TYPE_CHOICES = ("upper", "lower", "dress")
STYLE_CHOICES = ("tucked_in", "untucked", "one_piece")

STYLE_WORDS = {
    "tucked_in": "tucked in",
    "untucked": "untucked",
    "one_piece": "one-piece",
}


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def resolve_lora_dir(path: str) -> tuple[Path, str]:
    path_obj = Path(path)
    if path_obj.is_file():
        return path_obj.parent, path_obj.name
    return path_obj, "pytorch_lora_weights.safetensors"


def parse_csv_choices(value: str, choices: tuple[str, ...], arg_name: str) -> list[str]:
    value = value.strip()
    if value.lower() == "all":
        return list(choices)

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"--{arg_name} must list at least one value.")

    invalid = [item for item in items if item not in choices]
    if invalid:
        raise ValueError(
            f"Invalid value(s) for --{arg_name}: {invalid}. "
            f"Allowed: {list(choices)} or 'all'."
        )
    return items


def build_body_prompt(gender: str, shape: str, height: str) -> str:
    return f"The person is a {height} and {shape} {gender}."


def build_prompt(
    gender: str,
    shape: str,
    height: str,
    length: str,
    garment_type: str,
    style: str,
) -> str:
    body_prompt = build_body_prompt(gender, shape, height).strip()
    cloth_prompt = f"The cloth is a {length} {garment_type} garment."
    wear_prompt = f"The wearing style is {STYLE_WORDS[style]}."
    return f"{body_prompt} {cloth_prompt} {wear_prompt}"


def make_prompt_jobs(args) -> list[dict]:
    if args.custom_prompt:
        return [
            {
                "gender": None,
                "shape": None,
                "height": None,
                "length": None,
                "garment_type": None,
                "style": None,
                "prompt": args.custom_prompt.strip(),
            }
        ]

    genders = parse_csv_choices(args.gender, GENDER_CHOICES, "gender")
    shapes = parse_csv_choices(args.shape, SHAPE_CHOICES, "shape")
    heights = parse_csv_choices(args.height, HEIGHT_CHOICES, "height")
    lengths = parse_csv_choices(args.length, LENGTH_CHOICES, "length")
    garment_types = parse_csv_choices(args.garment_type, GARMENT_TYPE_CHOICES, "garment_type")
    styles = parse_csv_choices(args.style, STYLE_CHOICES, "style")

    jobs: list[dict] = []
    for gender, shape, height, length, garment_type, style in product(
        genders, shapes, heights, lengths, garment_types, styles
    ):
        jobs.append(
            {
                "gender": gender,
                "shape": shape,
                "height": height,
                "length": length,
                "garment_type": garment_type,
                "style": style,
                "prompt": build_prompt(gender, shape, height, length, garment_type, style),
            }
        )
    if not jobs:
        raise ValueError("No prompt jobs generated. Check prompt option combinations.")
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run single-sample FLUX-Kontext inference with long_vton prompts, "
            "transformer-only fitting LoRA, and optional texture LoRA. "
            "Prompt attributes accept comma-separated values to form a grid."
        )
    )
    parser.add_argument("--person_image", "--instance_image", "--img_path", dest="person_image", required=True)
    parser.add_argument("--reference_image", required=True)
    parser.add_argument(
        "--fitting_lora_dir",
        "--checkpoint_dir",
        dest="fitting_lora_dir",
        type=str,
        default=DEFAULT_FITTING_LORA,
    )
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
    parser.add_argument("--output_dir", type=str, default=cfg_path("outputs", "demo"))
    parser.add_argument("--output_prefix", type=str, default="vton")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--auto_resize", action="store_true")
    parser.add_argument(
        "--gender",
        type=str,
        default="female",
        help=f"Comma-separated values or 'all'. Choices: {list(GENDER_CHOICES)}.",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="slim",
        help=f"Comma-separated values or 'all'. Choices: {list(SHAPE_CHOICES)}.",
    )
    parser.add_argument(
        "--height",
        type=str,
        default="medium-tall",
        help=f"Comma-separated values or 'all'. Choices: {list(HEIGHT_CHOICES)}.",
    )
    parser.add_argument(
        "--length",
        type=str,
        default="short-length",
        help=f"Comma-separated values or 'all'. Choices: {list(LENGTH_CHOICES)}.",
    )
    parser.add_argument(
        "--garment_type",
        type=str,
        default="upper",
        help=f"Comma-separated values or 'all'. Choices: {list(GARMENT_TYPE_CHOICES)}.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="untucked",
        help=f"Comma-separated values or 'all'. Choices: {list(STYLE_CHOICES)}.",
    )
    parser.add_argument(
        "--custom_prompt",
        type=str,
        default=None,
        help="If set, ignore prompt grid options and run this exact prompt once.",
    )
    return parser.parse_args()


def load_pipeline(args):
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    flux_kontext_model_id = flux_model_id()
    longclip_repo_id = longclip_model_id()
    revision = "main"
    lora_dir, weight_name = resolve_lora_dir(args.fitting_lora_dir)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        flux_kontext_model_id, subfolder="scheduler", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        longclip_repo_id,
        torch_dtype=dtype,
    )
    tokenizer = CLIPTokenizer.from_pretrained(longclip_repo_id)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        flux_kontext_model_id,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
        revision=revision,
    )
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        flux_kontext_model_id, subfolder="tokenizer_2", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(
        flux_kontext_model_id, subfolder="vae", torch_dtype=dtype, revision=revision
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        flux_kontext_model_id,
        subfolder="transformer",
        torch_dtype=dtype,
        revision=revision,
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
    pipe.to(args.device, dtype=dtype)
    return pipe


def output_stem(args, job: dict, index: int) -> str:
    if job["gender"] is None:
        return slugify(f"{args.output_prefix}_{index:03d}_{job['prompt']}")[:120] or f"{args.output_prefix}_{index:03d}"
    return slugify(
        "_".join(
            [
                args.output_prefix,
                f"{index:03d}",
                job["gender"],
                job["shape"],
                job["height"],
                job["length"],
                job["garment_type"],
                job["style"],
                f"seed{args.seed}",
            ]
        )
    )


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = make_prompt_jobs(args)
    print(f"Running {len(jobs)} prompt job(s). Output dir: {output_dir}")

    pipe = load_pipeline(args)
    person_image = load_image(args.person_image)
    reference_image = load_image(args.reference_image)

    manifest_path = output_dir / "results.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, job in enumerate(jobs):
            print(f"[{index + 1}/{len(jobs)}] {job['prompt']}")

            with torch.inference_mode():
                image = pipe(
                    multiple_images=[(person_image, reference_image)],
                    prompt=job["prompt"],
                    width=args.width,
                    height=args.height,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=torch.Generator(args.device).manual_seed(args.seed + index),
                    num_images_per_prompt=1,
                    _auto_resize=args.auto_resize,
                ).images[0]

            out_path = output_dir / f"{output_stem(args, job, index)}.png"
            image.save(out_path, format="PNG", optimize=True, compress_level=9)

            row = {
                "output": str(out_path),
                "person_image": args.person_image,
                "reference_image": args.reference_image,
                "fitting_lora_dir": args.fitting_lora_dir,
                "texture_lora_dir": None if args.no_texture_lora else args.texture_lora_dir,
                "fitting_lora_weight": args.fitting_lora_weight,
                "texture_lora_weight": args.texture_lora_weight,
                "seed": args.seed + index,
                "width": args.width,
                "height": args.height,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                **job,
            }
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Saved: {out_path}")

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main(parse_args())
