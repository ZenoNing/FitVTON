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

GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
NUM_SHARDS="${#GPU_ARRAY[@]}"
LOG_DIR="${LOG_DIR:-$FITVTON_ROOT/outputs/logs/pseudo_images}"
DRESSCODE_ROOT="${DRESSCODE_ROOT:-$(fitvton_cfg datasets.dresscode_root)}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-$(fitvton_cfg checkpoints.root)}"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --pairs_file "$FITVTON_ROOT/pseudo.txt"
  --dataset_root "$DRESSCODE_ROOT"
  --output_root "$(fitvton_cfg outputs.pseudo_images)"
  --fitting_lora_dir "$CHECKPOINTS_ROOT/default_lora_weights.safetensors"
  --num_inference_steps "${NUM_INFERENCE_STEPS:-30}"
  --guidance_scale "${GUIDANCE_SCALE:-1.0}"
  --seed "${SEED:-0}"
)

if [[ -n "${LIMIT_PER_SHARD:-}" ]]; then
  COMMON_ARGS+=(--limit "$LIMIT_PER_SHARD")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

echo "Generating pseudo images with ${NUM_SHARDS} shard(s): ${GPUS}"

pids=()
for shard in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$shard]}"
  log="$LOG_DIR/shard_${shard}_gpu_${gpu}.log"
  echo "Starting shard ${shard}/${NUM_SHARDS} on GPU ${gpu}; log=${log}"
  CUDA_VISIBLE_DEVICES="$gpu" python generate_pseudo_images.py \
    "${COMMON_ARGS[@]}" \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$shard" \
    > "$log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" != "0" ]]; then
  echo "At least one pseudo-generation shard failed. Check logs in ${LOG_DIR}." >&2
  exit 1
fi

python - <<PY
from pathlib import Path
import csv

from system_config import cfg_path

pseudo = Path("${FITVTON_ROOT}") / "pseudo.txt"
root = Path(cfg_path("outputs", "pseudo_images"))
rows = list(csv.reader(pseudo.open("r", encoding="utf-8"), delimiter="\t"))
missing = [row[6] for row in rows if len(row) == 7 and not (root / row[6]).exists()]
print(f"pseudo rows={len(rows)} generated={len(rows) - len(missing)} missing={len(missing)}")
if missing:
    print("first missing:", missing[:10])
    raise SystemExit(1)
PY

echo "Pseudo generation finished."
