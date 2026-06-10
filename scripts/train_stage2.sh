#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export CUDA_HOME="$CONDA_PREFIX"
  export CUDA_PATH="$CONDA_PREFIX"
  export PATH="$CONDA_PREFIX/bin:${PATH}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FITVTON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$FITVTON_ROOT"

fitvton_cfg() {
  python "$FITVTON_ROOT/system_config.py" "$1"
}

NUM_GPUS="${NUM_GPUS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$(fitvton_cfg outputs.texture_lora)}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-$(fitvton_cfg checkpoints.root)}"
FITTING_LORA_DIR="${FITTING_LORA_DIR:-$CHECKPOINTS_ROOT/default_lora_weights.safetensors}"
DRESSCODE_ROOT="${DRESSCODE_ROOT:-$(fitvton_cfg datasets.dresscode_root)}"
REPORT_TO="${REPORT_TO:-tensorboard}"

MAX_STEP_ARGS=()
if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  MAX_STEP_ARGS=(--max_train_steps "$MAX_TRAIN_STEPS")
fi

accelerate launch --num_processes "$NUM_GPUS" train_texture_lora.py \
  --output_dir "$OUTPUT_DIR" \
  --revision main \
  --vae_encode_mode sample \
  --dresscode_root "$DRESSCODE_ROOT" \
  --pairs_file "$FITVTON_ROOT/second.txt" \
  --pseudo_data_root "$(fitvton_cfg outputs.pseudo_images)" \
  --fitting_lora_dir "$FITTING_LORA_DIR" \
  --rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.0 \
  --checkpointing_steps 200 \
  --checkpoints_total_limit 3 \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-5}" \
  "${MAX_STEP_ARGS[@]}" \
  --guidance_scale 1.0 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 500 \
  --lr_num_cycles 1 \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --report_to "$REPORT_TO" \
  --mixed_precision bf16 \
  --use_8bit_adam \
  --local_rank -1 \
  --texture_loss_type scharr \
  --texture_loss_weight 0.1 \
  --skip_fuse_fitting_lora
