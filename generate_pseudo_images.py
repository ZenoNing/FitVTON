#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers.utils.loading_utils import load_image

from inference_demo import load_pipeline
from system_config import cfg_path, fitting_lora_checkpoint, pseudo_pairs_file


TYPE_TO_DIR = {
    "0": "upper_body",
    "1": "lower_body",
    "2": "dresses",
}
TYPE_TO_PROMPT = {
    "0": "upper",
    "1": "lower",
    "2": "dress",
}


def normalize_length(length: str) -> str:
    length = length.strip()
    if length in {"short", "short-length"}:
        return "short-length"
    if length in {"long", "long-length"}:
        return "long-length"
    return length


def wearing_style_word(style: str) -> str:
    if style in {"tucked", "tucked_in"}:
        return "tucked in"
    if style == "untucked":
        return "untucked"
    if style == "one_piece":
        return "one-piece"
    return style.replace("_", " ")


def build_prompt(gender: str, style: str, length: str, type_id: str) -> str:
    return " ".join(
        [
            f"The person is a tall and slim {gender}.",
            f"The cloth is a {normalize_length(length)} {TYPE_TO_PROMPT[type_id]} garment.",
            f"The wearing style is {wearing_style_word(style)}.",
        ]
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, cols in enumerate(reader, 1):
            if not cols:
                continue
            if len(cols) != 7:
                raise ValueError(f"{path}:{line_no}: expected 7 columns, got {len(cols)}")
            person_image, reference_image, gender, style, length, type_id, output = cols
            if type_id not in TYPE_TO_DIR:
                raise ValueError(f"{path}:{line_no}: unknown type_id {type_id!r}")
            rows.append(
                {
                    "line_no": str(line_no),
                    "person_image": person_image,
                    "reference_image": reference_image,
                    "gender": gender,
                    "style": style,
                    "length": length,
                    "type_id": type_id,
                    "output": output,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate second-stage pseudo images from pseudo.txt.")
    parser.add_argument(
        "--pairs_file",
        "--pseudo",
        dest="pairs_file",
        type=Path,
        default=Path(pseudo_pairs_file()),
        help="TSV listing person/reference rows for pseudo generation.",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path(os.environ.get("DRESSCODE_ROOT", cfg_path("datasets", "dresscode_root"))),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(cfg_path("outputs", "pseudo_images")),
    )
    parser.add_argument(
        "--fitting_lora_dir",
        "--checkpoint_dir",
        dest="fitting_lora_dir",
        default=fitting_lora_checkpoint(),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--auto_resize", action="store_true")
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must be in [0, --num_shards)")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for idx, row in enumerate(read_rows(args.pairs_file))
        if idx % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    pipe_args = SimpleNamespace(
        fitting_lora_dir=args.fitting_lora_dir,
        device=args.device,
        dtype=args.dtype,
        texture_lora_dir=None,
        no_texture_lora=True,
        fitting_lora_weight=1.0,
        texture_lora_weight=0.0,
    )
    pipe = load_pipeline(pipe_args)

    manifest_path = args.output_root / f"manifest_shard{args.shard_index:02d}.jsonl"
    generated = 0
    skipped = 0
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for row in rows:
            category = TYPE_TO_DIR[row["type_id"]]
            person_image_path = args.dataset_root / category / "images" / row["person_image"]
            reference_path = args.dataset_root / category / "images" / row["reference_image"]
            output_path = args.output_root / row["output"]

            if output_path.exists() and not args.overwrite:
                skipped += 1
                print(f"skip existing: {output_path}", flush=True)
                continue
            if not person_image_path.exists():
                raise FileNotFoundError(f"missing person image: {person_image_path}")
            if not reference_path.exists():
                raise FileNotFoundError(f"missing reference image: {reference_path}")

            prompt = build_prompt(
                gender=row["gender"],
                style=row["style"],
                length=row["length"],
                type_id=row["type_id"],
            )
            print(
                f"[shard {args.shard_index}/{args.num_shards}] line={row['line_no']} "
                f"output={row['output']} prompt={prompt}",
                flush=True,
            )
            image = pipe(
                multiple_images=[(load_image(str(person_image_path)), load_image(str(reference_path)))],
                prompt=prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(args.device).manual_seed(args.seed),
                num_images_per_prompt=1,
                _auto_resize=args.auto_resize,
            ).images[0]
            image.save(output_path, format="JPEG", quality=95, optimize=True)
            manifest.write(
                json.dumps(
                    {
                        **row,
                        "person_image_path": str(person_image_path),
                        "reference_path": str(reference_path),
                        "output_path": str(output_path),
                        "prompt": prompt,
                        "seed": args.seed,
                        "num_inference_steps": args.num_inference_steps,
                        "guidance_scale": args.guidance_scale,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            manifest.flush()
            generated += 1
            print(f"saved: {output_path}", flush=True)

    print(f"Done. generated={generated} skipped={skipped} manifest={manifest_path}")


if __name__ == "__main__":
    main()
