#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL
from tqdm.auto import tqdm

from dataset import GarmentCodeVTONDataset
from system_config import cfg_path, flux_model_id
from head import GarmentMaskHead

logger = get_logger(__name__)


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.clamp(0.0, 1.0)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def compute_single_mask_loss(final_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    target_mask = target_mask.to(final_logits.device, dtype=final_logits.dtype)
    bce = nn.BCEWithLogitsLoss()(final_logits, target_mask)
    dice = dice_loss_with_logits(final_logits, target_mask)
    return bce + dice


def collate_fn(examples):
    gt_imgs = [ex["gt_image"] for ex in examples]
    body_masks = [ex["body_mask_image"] for ex in examples]
    garment_masks = [ex["garment_mask_image"] for ex in examples]

    return {
        "gt_images": torch.stack(gt_imgs),
        "body_mask_images": torch.stack(body_masks),
        "garment_mask_images": torch.stack(garment_masks),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train body/garment mask heads (supervised).")
    parser.add_argument("--garmentcode_root", "--instance_data_dir", dest="garmentcode_root", type=str, required=True, help="GarmentCodeVTON dataset root")
    parser.add_argument(
        "--max_pairs_per_group",
        type=int,
        default=None,
        help="Cap pairs per (gender, person, pose) group.",
    )
    parser.add_argument(
        "--include_genders",
        type=str,
        default="female,male",
        help="Comma-separated genders to include.",
    )
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="all",
        choices=["all", "dress_to_dress"],
        help="Which triplets to build.",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="long_vton",
        choices=["long_vton"],
        help="Prompt format to use (only long_vton is supported).",
    )
    parser.add_argument(
        "--training_stage",
        "--curriculum_stage",
        dest="training_stage",
        type=str,
        default="all",
        choices=["all", "cloth_balanced", "wearing_two_piece", "shape_balanced", "wearing_all_style"],
        help="Stage-specific filtering/stratified sampling.",
    )
    parser.add_argument("--output_dir", type=str, default=cfg_path("outputs", "maskhead"), help="Output directory to save mask heads")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None, help="If set, override num_train_epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument(
        "--vae_encode_mode",
        type=str,
        default="sample",
        choices=["sample", "mode"],
        help="Must match train_fitting_lora.py so VAE round-trips stay consistent.",
    )
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=Path(args.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    flux_kontext_model_id = flux_model_id()
    revision = "main"

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = AutoencoderKL.from_pretrained(flux_kontext_model_id, subfolder="vae", revision=revision)
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor

    mask_input_channels = 3
    mask_out_size = (1024, 768)

    body_mask_head = GarmentMaskHead(
        in_channels=mask_input_channels,
        out_size=mask_out_size,
    )
    garment_mask_head = GarmentMaskHead(
        in_channels=mask_input_channels,
        out_size=mask_out_size,
    )
    body_mask_head.mask_role = "body"
    garment_mask_head.mask_role = "garment"

    body_mask_head.train().requires_grad_(True)
    garment_mask_head.train().requires_grad_(True)

    optimizer = torch.optim.AdamW(
        list(body_mask_head.parameters()) + list(garment_mask_head.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )

    dataset_kwargs = dict(
        data_root=args.garmentcode_root,
        transform=True,
        include_genders=args.include_genders,
        pair_mode=args.pair_mode,
        prompt_style=args.prompt_style,
        training_stage=args.training_stage,
    )
    if args.max_pairs_per_group is not None:
        dataset_kwargs["max_pairs_per_group"] = args.max_pairs_per_group
        dataset_kwargs["cap_seed"] = args.seed
    train_dataset = GarmentCodeVTONDataset(**dataset_kwargs)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )

    body_mask_head, garment_mask_head, optimizer, train_dataloader = accelerator.prepare(
        body_mask_head, garment_mask_head, optimizer, train_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="Steps")

    for epoch in range(args.num_train_epochs):
        body_mask_head.train()
        garment_mask_head.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate([body_mask_head, garment_mask_head]):
                gt_images = batch["gt_images"].to(accelerator.device, dtype=weight_dtype)
                body_mask_gt = batch["body_mask_images"].to(accelerator.device)
                garment_mask_gt = batch["garment_mask_images"].to(accelerator.device)

                with torch.no_grad():
                    if args.vae_encode_mode == "sample":
                        latents = vae.encode(gt_images).latent_dist.sample()
                    else:
                        latents = vae.encode(gt_images).latent_dist.mode()

                    z_0 = (latents - vae_config_shift_factor) * vae_config_scaling_factor
                    mask_latents = z_0 / vae_config_scaling_factor + vae_config_shift_factor
                    mask_pixels = vae.decode(mask_latents).sample.to(dtype=weight_dtype)

                body_logits = body_mask_head(mask_pixels)
                garment_logits = garment_mask_head(mask_pixels)

                body_loss = compute_single_mask_loss(body_logits, body_mask_gt)
                garment_loss = compute_single_mask_loss(garment_logits, garment_mask_gt)
                loss = body_loss + garment_loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=loss.detach().float().item(),
                    body=body_loss.detach().float().item(),
                    garment=garment_loss.detach().float().item(),
                )

                if accelerator.is_main_process and args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(accelerator.unwrap_model(body_mask_head).state_dict(), ckpt_dir / "body_mask_head.pth")
                    torch.save(
                        accelerator.unwrap_model(garment_mask_head).state_dict(), ckpt_dir / "garment_mask_head.pth"
                    )

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(body_mask_head).state_dict(), Path(args.output_dir) / "body_mask_head.pth")
        torch.save(
            accelerator.unwrap_model(garment_mask_head).state_dict(), Path(args.output_dir) / "garment_mask_head.pth"
        )
        logger.info(f"Saved final mask heads to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
