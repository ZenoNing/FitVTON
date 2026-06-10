#!/usr/bin/env bash
set -euo pipefail

# Six-stage V3 curriculum:
#   1) shape-heavy dress_to_dress
#   2) cloth-type-balanced full data
#   3) wearing-style two-piece focused
#   4) shape-balanced full data
#   5) cloth-type-balanced full data
#   6) wearing-style focused with one-piece retained
#
# All stages:
#   - transformer LoRA only (text encoders frozen)
#   - prompt_style=long_vton
#   - max_pairs_per_group=120
#   - 1 epoch per stage
#   - cosine_with_restarts with lr_num_cycles=1 per stage
#
# Usage:
#   conda activate flux
#   bash scripts/train_stage1.sh
#
# Optional:
#   SKIP_MASK_HEAD=1            skip mask-head pretraining even if weights are missing
#   NUM_GPUS=4 TRAIN_BATCH_SIZE=2 CAP=120

# Prefer the active conda env CUDA toolkit; do not override with system CUDA.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export CUDA_HOME="$CONDA_PREFIX"
  export CUDA_PATH="$CONDA_PREFIX"
  export PATH="$CONDA_PREFIX/bin:${PATH}"
else
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
  export CUDA_PATH="$CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FITVTON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$FITVTON_ROOT"

fitvton_cfg() {
  python "$FITVTON_ROOT/system_config.py" "$1"
}

DATA_ROOT="${DATA_ROOT:-$(fitvton_cfg datasets.garmentcode_root)}"
MASK_HEAD_DIR="${MASK_HEAD_DIR:-$(fitvton_cfg outputs.maskhead)}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$(fitvton_cfg outputs.fitting_lora)}"

NUM_GPUS=${NUM_GPUS:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-1}
CAP=${CAP:-120}
SEED=${SEED:-0}

mkdir -p "$BASE_OUTPUT_DIR"

dataset_steps() {
  local pair_mode="$1"
  local curriculum_stage="$2"
  "$PYTHON" - "$DATA_ROOT" "$pair_mode" "$curriculum_stage" "$CAP" "$SEED" "$NUM_GPUS" "$TRAIN_BATCH_SIZE" "$GRAD_ACCUM" <<'PY'
import math
import sys

from dataset import GarmentCodeVTONDataset

data_root, pair_mode, training_stage = sys.argv[1], sys.argv[2], sys.argv[3]
cap, seed = int(sys.argv[4]), int(sys.argv[5])
num_gpus, batch_size, grad_accum = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])

ds = GarmentCodeVTONDataset(
    data_root,
    include_genders="female,male",
    pair_mode=pair_mode,
    prompt_style="long_vton",
    training_stage=training_stage,
    max_pairs_per_group=cap,
    cap_seed=seed,
)
steps = math.ceil(len(ds) / (num_gpus * batch_size * grad_accum))
print(steps)
PY
}

if [[ "${SKIP_MASK_HEAD:-0}" == "1" ]]; then
  echo "SKIP_MASK_HEAD=1: skipping mask-head training (using $MASK_HEAD_DIR if present)"
elif [[ ! -f "$MASK_HEAD_DIR/body_mask_head.pth" || ! -f "$MASK_HEAD_DIR/garment_mask_head.pth" ]]; then
  echo "Mask-head weights not found in $MASK_HEAD_DIR; training them first."
  accelerate launch train_maskhead.py \
    --output_dir "$MASK_HEAD_DIR" \
    --garmentcode_root "$DATA_ROOT" \
    --include_genders female,male \
    --pair_mode all \
    --training_stage all \
    --max_pairs_per_group "$CAP" \
    --vae_encode_mode sample \
    --checkpointing_steps 1000 \
    --train_batch_size 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --dataloader_num_workers 0
else
  echo "Using existing mask-head weights from $MASK_HEAD_DIR"
fi

run_stage() {
  local stage_id="$1"
  local stage_name="$2"
  local pair_mode="$3"
  local curriculum_stage="$4"
  local init_lora_dir="${5:-}"
  local output_dir="$BASE_OUTPUT_DIR/${stage_id}_${stage_name}"
  local steps
  steps="$(dataset_steps "$pair_mode" "$curriculum_stage")"

  echo
  echo "===== ${stage_id}: ${stage_name} ====="
  echo "pair_mode=${pair_mode}, curriculum_stage=${curriculum_stage}, max_train_steps=${steps}"
  echo "output_dir=${output_dir}"
  if [[ -n "$init_lora_dir" ]]; then
    echo "init_lora_dir=${init_lora_dir}"
  fi

  local init_args=()
  if [[ -n "$init_lora_dir" ]]; then
    init_args=(--init_lora_dir "$init_lora_dir")
  fi

  accelerate launch train_fitting_lora.py \
    --output_dir "$output_dir" \
    --revision main \
    --vae_encode_mode sample \
    --pretrained_mask_head_dir "$MASK_HEAD_DIR" \
    --garmentcode_root "$DATA_ROOT" \
    --include_genders female,male \
    --pair_mode "$pair_mode" \
    --training_stage "$curriculum_stage" \
    --max_pairs_per_group "$CAP" \
    --rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.0 \
    --checkpointing_steps 1000 \
    --checkpoints_total_limit 3 \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --max_train_steps "$steps" \
    --guidance_scale 1.0 \
    --gradient_checkpointing \
    --learning_rate 1e-4 \
    --lr_scheduler cosine_with_restarts \
    --lr_warmup_steps 500 \
    --lr_num_cycles 1 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --report_to wandb \
    --mixed_precision bf16 \
    --use_8bit_adam \
    --local_rank -1 \
    --save_final_training_state \
    "${init_args[@]}"
}

STAGE1_DIR="$BASE_OUTPUT_DIR/stage1_shape_dress"
STAGE2_DIR="$BASE_OUTPUT_DIR/stage2_cloth_balanced"
STAGE3_DIR="$BASE_OUTPUT_DIR/stage3_wearing_two_piece"
STAGE4_DIR="$BASE_OUTPUT_DIR/stage4_shape_full"
STAGE5_DIR="$BASE_OUTPUT_DIR/stage5_cloth_balanced"

run_stage stage1 shape_dress dress_to_dress all
run_stage stage2 cloth_balanced all cloth_balanced "$STAGE1_DIR"
run_stage stage3 wearing_two_piece all wearing_two_piece "$STAGE2_DIR"
run_stage stage4 shape_full all shape_balanced "$STAGE3_DIR"
run_stage stage5 cloth_balanced all cloth_balanced "$STAGE4_DIR"
run_stage stage6 wearing_all_style all wearing_all_style "$STAGE5_DIR"

echo
echo "Curriculum training finished."
echo "Final LoRA directory: $BASE_OUTPUT_DIR/stage6_wearing_all_style"
