#!/usr/bin/env python
# coding=utf-8
"""Train stage-2 FitVTON texture LoRA with frozen fitting LoRA and optional texture loss."""

import argparse
import copy
import logging
import math
import os
import shutil
from pathlib import Path

import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate.state import AcceleratorState
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from dataset import DressCodeDataset
from system_config import cfg_path, fitting_lora_checkpoint, flux_model_id, longclip_model_id, second_stage_pairs_file
from tqdm.auto import tqdm
from transformers.models.clip import CLIPTextModel, CLIPTokenizer
from transformers.models.t5 import T5EncoderModel, T5TokenizerFast

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_kontext_multiple_images import FluxKontextPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, is_wandb_available
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

check_min_version("0.35.1")

logger = get_logger(__name__)

def resolve_lora_location(path: str) -> tuple[str, str | None]:
    path_obj = Path(path)
    if path_obj.is_file():
        return str(path_obj.parent), path_obj.name
    return str(path_obj), None


if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train FitVTON stage-2 texture LoRA on DressCode pseudo data.")
    parser.add_argument("--revision", type=str, default=None, help="Base model revision.")
    parser.add_argument(
        "--vae_encode_mode",
        type=str,
        default="sample",
        choices=["sample", "mode"],
        help="VAE encoding mode.",
    )
    parser.add_argument("--dresscode_root", "--instance_data_dir", "--dresscode_data_root", dest="dresscode_root", type=str, required=True, help="DressCode dataset root.")
    parser.add_argument(
        "--pairs_file", "--second_stage_pairs_file", dest="pairs_file", type=str, default=second_stage_pairs_file(), help="7-column TSV for pseudo/reference/gt training pairs.",
    )
    parser.add_argument(
        "--pseudo_data_root",
        type=str,
        default=cfg_path("outputs", "pseudo_images"),
        help="Directory containing generated pseudo images.",
    )
    parser.add_argument(
        "--dresscode_data_root",
        type=str,
        default=None,
        help="Deprecated alias for --dresscode_root.",
    )
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default=cfg_path("outputs", "texture_lora"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--fitting_lora_dir",
        type=str,
        default=fitting_lora_checkpoint(),
        help="Pretrained fitting LoRA checkpoint file or directory.",
    )
    parser.add_argument("--texture_adapter_name", type=str, default="texture")
    parser.add_argument(
        "--skip_fuse_fitting_lora",
        action="store_true",
        help="Keep fitting LoRA as a separate frozen adapter instead of fusing into base weights.",
    )
    parser.add_argument(
        "--texture_loss_type",
        type=str,
        default="scharr",
        choices=["none", "laplacian", "scharr", "dog"],
    )
    parser.add_argument("--texture_loss_weight", type=float, default=0.05)
    parser.add_argument("--texture_loss_every", type=int, default=1)
    parser.add_argument("--dog_sigma1", type=float, default=0.8)
    parser.add_argument("--dog_sigma2", type=float, default=1.6)
    parser.add_argument("--dog_kernel_size", type=int, default=9)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--upcast_before_saving", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def collate_fn(examples):

    pseudo_imgs = [example["pseudo_image"] for example in examples]
    reference_imgs = [example["reference_image"] for example in examples]
    gt_imgs = [example["gt_image"] for example in examples]
    prompts = [example["prompt"] for example in examples]

    batch = {
        "pseudo_images": torch.stack(pseudo_imgs),
        "reference_images": torch.stack(reference_imgs),
        "gt_images": torch.stack(gt_imgs),
        "prompts": prompts,
    }
    return batch


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding=True,
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding=True,
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding=True,
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids

def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    flux_kontext_model_id = flux_model_id()
    longclip_repo_id = longclip_model_id()
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    clip_tokenizer = CLIPTokenizer.from_pretrained(longclip_repo_id, revision=args.revision)
    t5_tokenizer = T5TokenizerFast.from_pretrained(flux_kontext_model_id, subfolder="tokenizer_2", revision=args.revision)
    clip_text_encoder = CLIPTextModel.from_pretrained(longclip_repo_id, revision=args.revision)
    t5_text_encoder = T5EncoderModel.from_pretrained(flux_kontext_model_id, subfolder="text_encoder_2", revision=args.revision)
    clip_text_encoder.requires_grad_(False)
    t5_text_encoder.requires_grad_(False)
    # Keep frozen text encoders on CPU; only prompt embeddings move to the training device.
    clip_text_encoder.to("cpu", dtype=torch.float32)
    t5_text_encoder.to("cpu", dtype=torch.float32)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(flux_kontext_model_id, subfolder="scheduler", revision=args.revision)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKL.from_pretrained(flux_kontext_model_id,subfolder="vae", revision=args.revision)
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    transformer = FluxTransformer2DModel.from_pretrained(flux_kontext_model_id, subfolder="transformer", revision=args.revision)
    transformer.requires_grad_(False)
    transformer.to(accelerator.device, dtype=weight_dtype)

    def _set_single_adapter_if_possible(model, adapter_name: str):
        try:
            if hasattr(model, "set_adapter"):
                model.set_adapter(adapter_name)
            elif hasattr(model, "set_adapters"):
                model.set_adapters(adapter_name)
        except Exception as e:
            logger.warning(f"Could not activate adapter '{adapter_name}' for {model.__class__.__name__}: {e}")

    def _fuse_peft_adapter_into_base(model, adapter_name: str, label: str):
        """Best-effort fuse/merge PEFT LoRA adapter weights into base weights and unload adapters."""
        _set_single_adapter_if_possible(model, adapter_name)

        if hasattr(model, "fuse_lora"):
            try:
                try:
                    model.fuse_lora(adapter_names=[adapter_name])
                except TypeError:
                    model.fuse_lora()
                logger.info(f"Fused adapter '{adapter_name}' into base weights for {label} via fuse_lora().")
                return model
            except Exception as e:
                logger.warning(f"fuse_lora() failed for {label}: {e}")

        logger.warning(
            f"No supported fusion method found for {label}. Continuing without fusing; "
            f"stage-2 may still behave like dual-adapter training."
        )

        if hasattr(model, "merge_and_unload"):
            try:
                fused = model.merge_and_unload()
                logger.info(f"Fused adapter '{adapter_name}' into base weights for {label} via merge_and_unload().")
                return fused
            except Exception as e:
                logger.warning(f"merge_and_unload() failed for {label}: {e}")

        if hasattr(model, "merge_adapter"):
            try:
                try:
                    model.merge_adapter(adapter_name)
                except TypeError:
                    model.merge_adapter()
                # Delete the merged adapter so a later set_adapter() does not unmerge fused weights.
                if hasattr(model, "delete_adapter"):
                    try:
                        model.delete_adapter(adapter_name)
                    except TypeError:
                        model.delete_adapter()
                elif hasattr(model, "delete_adapters"):
                    try:
                        model.delete_adapters(adapter_name)
                    except TypeError:
                        model.delete_adapters()

                logger.info(
                    f"Fused adapter '{adapter_name}' into base weights for {label} via merge_adapter() + delete_adapter(s)."
                )
                return model
            except Exception as e:
                logger.warning(f"merge_adapter() failed for {label}: {e}")


        return model

    def _to_luma(img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] == 1:
            return img
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    def _conv2d_same(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        kh, kw = kernel.shape[-2:]
        pad_h = kh // 2
        pad_w = kw // 2
        return torch.nn.functional.conv2d(x, kernel, padding=(pad_h, pad_w))

    def _scharr_edges(gray: torch.Tensor) -> torch.Tensor:
        kx = torch.tensor(
            [[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[3.0, 10.0, 3.0], [0.0, 0.0, 0.0], [-3.0, -10.0, -3.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        gx = _conv2d_same(gray, kx)
        gy = _conv2d_same(gray, ky)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def _laplacian(gray: torch.Tensor) -> torch.Tensor:
        k = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        return _conv2d_same(gray, k)

    def _gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
        if kernel_size % 2 == 0:
            kernel_size += 1
        ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size // 2)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma + 1e-8))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def _dog(gray: torch.Tensor, sigma1: float, sigma2: float, kernel_size: int) -> torch.Tensor:
        k1 = _gaussian_kernel2d(kernel_size, sigma1, gray.device, gray.dtype)
        k2 = _gaussian_kernel2d(kernel_size, sigma2, gray.device, gray.dtype)
        b1 = _conv2d_same(gray, k1)
        b2 = _conv2d_same(gray, k2)
        return b1 - b2

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    target_modules = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ]

    # fitting_target_modules must match stage-1 exactly or weights are silently dropped.
    fitting_target_modules = [
        "context_embedder",
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    transformer_fitting_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=fitting_target_modules,
    )
    transformer.add_adapter(transformer_fitting_lora_config)

    text_lora_config_clip = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    text_lora_config_t5 = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=["q", "k", "v", "o"],
    )

    if args.fitting_lora_dir is not None and os.path.exists(args.fitting_lora_dir):
        lora_dir, weight_name = resolve_lora_location(args.fitting_lora_dir)
        lora_kwargs = {"weight_name": weight_name} if weight_name else {}
        lora_state_dict = FluxKontextPipeline.lora_state_dict(lora_dir, **lora_kwargs)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")

        if any(k.startswith("text_encoder.") for k in lora_state_dict):
            clip_text_encoder.add_adapter(text_lora_config_clip)
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=clip_text_encoder)
        else:
            logger.info("No pretrained text_encoder LoRA keys found; keeping CLIP text encoder frozen without adapters.")

        t5_text_encoder_lora_path = os.path.join(lora_dir, "text_encoder_2_lora.pth")
        if os.path.exists(t5_text_encoder_lora_path):
            t5_text_encoder.add_adapter(text_lora_config_t5)
            te2_state_dict = torch.load(t5_text_encoder_lora_path, map_location="cpu")
            set_peft_model_state_dict(t5_text_encoder, te2_state_dict, adapter_name="default")
            logger.info(f"Loaded pretrained text_encoder_2 LoRA from {t5_text_encoder_lora_path}")
        else:
            logger.info(f"No pretrained text_encoder_2 LoRA found at {t5_text_encoder_lora_path}")

        logger.info(f"Loaded pretrained fitting LoRA from {args.fitting_lora_dir}")
    else:
        logger.warning(f"fitting_lora_dir not found or not set: {args.fitting_lora_dir}")

    _set_single_adapter_if_possible(transformer, "default")

    if not args.skip_fuse_fitting_lora:
        transformer = _fuse_peft_adapter_into_base(transformer, adapter_name="default", label="transformer")
        transformer.to(accelerator.device, dtype=weight_dtype)

    for p in transformer.parameters():
        p.requires_grad_(False)
    for p in clip_text_encoder.parameters():
        p.requires_grad_(False)
    for p in t5_text_encoder.parameters():
        p.requires_grad_(False)

    transformer_texture_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_texture_lora_config, adapter_name=args.texture_adapter_name)

    for name, param in transformer.named_parameters():
        if args.texture_adapter_name in name:
            param.requires_grad_(True)

    if args.skip_fuse_fitting_lora:
        if hasattr(transformer, "set_adapters"):
            transformer.set_adapters(["default", args.texture_adapter_name])
        elif hasattr(transformer, "set_adapter"):
            transformer.set_adapter(["default", args.texture_adapter_name])
        logger.info(
            f"skip_fuse_fitting_lora: activated both 'default' and '{args.texture_adapter_name}' adapters."
        )
    else:
        _set_single_adapter_if_possible(transformer, args.texture_adapter_name)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    try:
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model, adapter_name=args.texture_adapter_name)
                    except TypeError:
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                weights.pop()

            FluxKontextPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=None,
                **_collate_lora_metadata(modules_to_save),
            )


    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxKontextPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            transformer_, transformer_state_dict, adapter_name=args.texture_adapter_name
        )
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.mixed_precision == "fp16":
            models = [transformer_]
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    params_to_optimize = [{"params": transformer_lora_parameters, "lr": args.learning_rate}]

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "To use 8-bit Adam, install bitsandbytes: `pip install bitsandbytes`."
            ) from exc
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    train_dataset = DressCodeDataset(
        dresscode_root=args.dresscode_root,
        pseudo_root=args.pseudo_data_root,
        pairs_file=args.pairs_file,
        transform=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    accelerator.wait_for_everyone()

    def encode_prompts_on_cpu(prompts):
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        clip_tokens = tokenize_prompt(clip_tokenizer, prompts, max_sequence_length=248)
        t5_tokens = tokenize_prompt(t5_tokenizer, prompts, max_sequence_length=args.max_sequence_length)
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders=[clip_text_encoder, t5_text_encoder],
                tokenizers=[None, None],
                text_input_ids_list=[clip_tokens, t5_tokens],
                max_sequence_length=args.max_sequence_length,
                device=torch.device("cpu"),
                prompt=prompts,
            )
        return (
            prompt_embeds.detach().cpu(),
            pooled_prompt_embeds.detach().cpu(),
            text_ids.detach().cpu(),
        )

    def build_prompt_embedding_cache(prompt_list=None):
        dataset_samples = getattr(train_dataset, "samples", None)
        if prompt_list is None and not dataset_samples:
            logger.info("Prompt embedding pre-cache skipped: dataset does not expose explicit samples.")
            return {}

        unique_prompts = sorted(prompt_list or {sample["prompt"] for sample in dataset_samples})
        logger.info(f"Pre-caching prompt embeddings for {len(unique_prompts)} unique prompt(s).")
        cache = {}
        if not unique_prompts:
            return cache

        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompts_on_cpu(unique_prompts)
        for idx, prompt in enumerate(
            tqdm(
                unique_prompts,
                desc="Caching prompt embeds",
                disable=not accelerator.is_local_main_process,
            )
        ):
            cache[prompt] = (
                prompt_embeds[idx : idx + 1],
                pooled_prompt_embeds[idx : idx + 1],
                text_ids,
            )
        return cache

    prompt_embedding_cache = build_prompt_embedding_cache()

    def get_prompt_embeddings(prompts):
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        if prompt_embedding_cache:
            missing = [prompt for prompt in prompts if prompt not in prompt_embedding_cache]
            if missing:
                rebuilt = build_prompt_embedding_cache(
                    sorted(set(prompt_embedding_cache.keys()).union(missing))
                )
                prompt_embedding_cache.clear()
                prompt_embedding_cache.update(rebuilt)

            cached = [prompt_embedding_cache[prompt] for prompt in prompts]
            prompt_embeds = torch.cat([item[0] for item in cached], dim=0)
            pooled_prompt_embeds = torch.cat([item[1] for item in cached], dim=0)
            text_ids = cached[0][2]
        else:
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompts_on_cpu(prompts)

        return (
            prompt_embeds.to(device=accelerator.device, dtype=weight_dtype, non_blocking=True),
            pooled_prompt_embeds.to(device=accelerator.device, dtype=weight_dtype, non_blocking=True),
            text_ids.to(device=accelerator.device, dtype=weight_dtype, non_blocking=True),
        )

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_name = "garmentcodevton-flux-kontext-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    has_guidance = unwrap_model(transformer).config.guidance_embeds
    for epoch in range(first_epoch, args.num_train_epochs):

        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                prompts = batch["prompts"]
                prompt_embeds, pooled_prompt_embeds, text_ids = get_prompt_embeddings(prompts)
                pseudo_images = batch["pseudo_images"].to(dtype=weight_dtype)
                garment_images = batch["reference_images"].to(dtype=weight_dtype)
                target_images = batch["gt_images"].to(dtype=weight_dtype)

                if args.vae_encode_mode == "sample":
                    target_latents = vae.encode(target_images).latent_dist.sample()
                    pseudo_cond_latents = vae.encode(pseudo_images).latent_dist.sample()
                    garment_cond_latents = vae.encode(garment_images).latent_dist.sample()
                else:
                    target_latents = vae.encode(target_images).latent_dist.mode()
                    pseudo_cond_latents = vae.encode(pseudo_images).latent_dist.mode()
                    garment_cond_latents = vae.encode(garment_images).latent_dist.mode()
                target_latents = (target_latents - vae_config_shift_factor) * vae_config_scaling_factor
                pseudo_cond_latents = (pseudo_cond_latents - vae_config_shift_factor) * vae_config_scaling_factor
                target_latents = target_latents.to(dtype=weight_dtype)
                pseudo_cond_latents = pseudo_cond_latents.to(dtype=weight_dtype)
                garment_cond_latents = (garment_cond_latents - vae_config_shift_factor) * vae_config_scaling_factor
                garment_cond_latents = garment_cond_latents.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    target_latents.shape[0],
                    target_latents.shape[2] // 2,
                    target_latents.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                pseudo_cond_latent_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    pseudo_cond_latents.shape[0],
                    pseudo_cond_latents.shape[2] // 2,
                    pseudo_cond_latents.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                pseudo_cond_latent_ids[..., 0] = 1
                latent_image_ids = torch.cat([latent_image_ids, pseudo_cond_latent_ids], dim=0)
                garment_cond_latent_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    garment_cond_latents.shape[0],
                    garment_cond_latents.shape[2] // 2,
                    garment_cond_latents.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                garment_cond_latent_ids[..., 0] = 2
                garment_cond_latent_ids[..., 2] += garment_cond_latents.shape[2] // 2
                latent_image_ids = torch.cat([latent_image_ids, garment_cond_latent_ids], dim=0)

                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=target_latents.device)
                sigmas = get_sigmas(timesteps, n_dim=target_latents.ndim, dtype=target_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * target_latents + sigmas * noise
                packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=noisy_model_input.shape[0],
                    num_channels_latents=noisy_model_input.shape[1],
                    height=noisy_model_input.shape[2],
                    width=noisy_model_input.shape[3],
                )
                orig_inp_shape = packed_noisy_model_input.shape
                packed_pseudo_cond = FluxKontextPipeline._pack_latents(
                    pseudo_cond_latents,
                    batch_size=pseudo_cond_latents.shape[0],
                    num_channels_latents=pseudo_cond_latents.shape[1],
                    height=pseudo_cond_latents.shape[2],
                    width=pseudo_cond_latents.shape[3],
                )
                packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_pseudo_cond], dim=1)
                packed_garment_cond = FluxKontextPipeline._pack_latents(
                    garment_cond_latents,
                    batch_size=garment_cond_latents.shape[0],
                    num_channels_latents=garment_cond_latents.shape[1],
                    height=garment_cond_latents.shape[2],
                    width=garment_cond_latents.shape[3],
                )
                packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_garment_cond], dim=1)

                guidance = None
                if has_guidance:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(target_latents.shape[0])

                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : orig_inp_shape[1]]
                model_pred = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=target_latents.shape[2] * vae_scale_factor,
                    width=target_latents.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - target_latents
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                texture_loss = None
                if (
                    args.texture_loss_type != "none"
                    and args.texture_loss_weight is not None
                    and args.texture_loss_weight > 0
                    and args.texture_loss_every is not None
                    and args.texture_loss_every >= 1
                    and (global_step % args.texture_loss_every == 0)
                ):
                    x0_pred = noise - model_pred
                    x0_pred_vae = x0_pred / vae_config_scaling_factor + vae_config_shift_factor

                    x0_pred_vae = x0_pred_vae.to(dtype=weight_dtype)

                    pred_img = vae.decode(x0_pred_vae).sample
                    gt_img_for_loss = target_images

                    pred_gray = _to_luma(pred_img.float())
                    gt_gray = _to_luma(gt_img_for_loss.float())

                    if args.texture_loss_type == "scharr":
                        pred_feat = _scharr_edges(pred_gray)
                        gt_feat = _scharr_edges(gt_gray)
                    elif args.texture_loss_type == "laplacian":
                        pred_feat = _laplacian(pred_gray)
                        gt_feat = _laplacian(gt_gray)
                    elif args.texture_loss_type == "dog":
                        pred_feat = _dog(pred_gray, args.dog_sigma1, args.dog_sigma2, args.dog_kernel_size)
                        gt_feat = _dog(gt_gray, args.dog_sigma1, args.dog_sigma2, args.dog_kernel_size)
                    else:
                        pred_feat = None
                        gt_feat = None

                    if pred_feat is not None:
                        texture_loss = torch.nn.functional.l1_loss(pred_feat, gt_feat)

                total_loss = loss if texture_loss is None else (loss + args.texture_loss_weight * texture_loss)

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if texture_loss is not None:
                logs["texture_loss"] = texture_loss.detach().item()
                logs["total_loss"] = total_loss.detach().item()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        modules_to_save = {}
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)

        try:
            transformer_lora_layers = get_peft_model_state_dict(transformer, adapter_name=args.texture_adapter_name)
        except TypeError:
            transformer_lora_layers = get_peft_model_state_dict(transformer)
        weight_name = f"pytorch_lora_weights.safetensors"
        modules_to_save["transformer"] = transformer
        FluxKontextPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=None,
            weight_name=weight_name,
            **_collate_lora_metadata(modules_to_save),
        )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
