#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" ]] && [[ "$(basename "$CONDA_PREFIX")" != "flux" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate flux
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FITVTON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$FITVTON_ROOT"

fitvton_cfg() {
  python "$FITVTON_ROOT/system_config.py" "$1"
}

GPUS="${GPUS:-0,1,2,3,4,5}"
FITTINGEFFECT_ROOT="${FITTINGEFFECT_ROOT:-$(fitvton_cfg datasets.fittingeffect_root)}"
TRIPLES_CSV="${TRIPLES_CSV:-$FITTINGEFFECT_ROOT/tryon_triples_all.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$(fitvton_cfg outputs.fittingeffect)}"
FEMALE_ROOT="${FEMALE_ROOT:-$FITTINGEFFECT_ROOT/female}"
MALE_ROOT="${MALE_ROOT:-$FITTINGEFFECT_ROOT/male}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-$(fitvton_cfg checkpoints.root)}"
FITTING_LORA_DIR="${FITTING_LORA_DIR:-$CHECKPOINTS_ROOT/default_lora_weights.safetensors}"
TEXTURE_LORA_DIR="${TEXTURE_LORA_DIR:-$CHECKPOINTS_ROOT/texture_lora_weights.safetensors}"
DEFAULT_LORA_WEIGHT="${DEFAULT_LORA_WEIGHT:-0.8}"
TEXTURE_LORA_WEIGHT="${TEXTURE_LORA_WEIGHT:-0.2}"
LOG_DIR="${LOG_DIR:-$FITVTON_ROOT/outputs/logs/fittingeffect_eval}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
NUM_SHARDS="${#GPU_ARRAY[@]}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

COMMON_ARGS=(
  --triples_csv "$TRIPLES_CSV"
  --female_root "$FEMALE_ROOT"
  --male_root "$MALE_ROOT"
  --cloth_ext jpg
  --flat_output
  --output_dir "$OUTPUT_DIR"
  --fitting_lora_dir "$FITTING_LORA_DIR"
  --texture_lora_dir "$TEXTURE_LORA_DIR"
  --fitting_lora_weight "$DEFAULT_LORA_WEIGHT"
  --texture_lora_weight "$TEXTURE_LORA_WEIGHT"
  --num_shards "$NUM_SHARDS"
)

if [[ "$DRY_RUN" == "1" ]]; then
  COMMON_ARGS+=(--dry_run)
fi

echo "FittingEffect eval with ${NUM_SHARDS} shard(s) on GPU(s): ${GPUS}"
echo "CSV=${TRIPLES_CSV}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

pids=()
for shard in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$shard]}"
  log="$LOG_DIR/shard_${shard}_gpu_${gpu}.log"
  echo "Starting shard ${shard}/${NUM_SHARDS} on GPU ${gpu}; log=${log}"
  CUDA_VISIBLE_DEVICES="$gpu" python inference_fittingeffect.py \
    "${COMMON_ARGS[@]}" \
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
  echo "At least one eval shard failed. Check logs in ${LOG_DIR}." >&2
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run finished."
  exit 0
fi

TRIPLES_CSV="$TRIPLES_CSV" OUTPUT_DIR="$OUTPUT_DIR" python - <<'PY'
import csv
import os
from pathlib import Path

csv_path = Path(os.environ["TRIPLES_CSV"])
output_dir = Path(os.environ["OUTPUT_DIR"])
rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
missing = []
for row in rows:
    out = output_dir / f"{row['source_person']}_{row['cloth']}.png"
    if not out.exists():
        missing.append(out.name)
print(f"expected={len(rows)} generated={len(rows) - len(missing)} missing={len(missing)}")
if missing:
    print("first missing:", missing[:10])
    raise SystemExit(1)
PY

echo "FittingEffect eval finished. Outputs: ${OUTPUT_DIR}"
