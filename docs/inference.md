# Inference

Activate the main environment first:

```bash
conda activate flux
```

## Single-Sample Demo

`inference_demo.py` supports Garment-Body Size prompt control. Prompt attributes accept comma-separated values to form a grid.

```bash
python inference_demo.py \
  --person_image /path/to/person.jpg \
  --reference_image /path/to/garment.jpg \
  --gender female \
  --shape slim \
  --height medium-tall \
  --length short-length \
  --garment_type upper \
  --style untucked
```

Both LoRAs are loaded by default:

- fitting LoRA weight: `0.8`
- texture LoRA weight: `0.2`

Pass `--no_texture_lora` to use the Stage I model only.

## FittingEffect3K Benchmark Generation

Configure `datasets.fittingeffect_root` and `checkpoints.root` in `system.json`, then run:

```bash
bash scripts/run_fittingeffect_eval.sh
```

The script shards the benchmark across GPUs and writes flat PNG outputs to `outputs.fittingeffect`.

Useful overrides:

```bash
GPUS=0,1 OUTPUT_DIR=/scratch/fitvton/fittingeffect \
  bash scripts/run_fittingeffect_eval.sh
```

`scripts/run_fittingeffect_eval.sh` also accepts:

| Variable | Meaning |
|----------|---------|
| `GPUS` | Comma-separated GPU list |
| `FITTINGEFFECT_ROOT` | Dataset root |
| `TRIPLES_CSV` | Triples CSV |
| `OUTPUT_DIR` | Prediction output directory |
| `CHECKPOINTS_ROOT` | Directory with LoRA weights |
| `FITTING_LORA_DIR` | Explicit fitting LoRA path |
| `TEXTURE_LORA_DIR` | Explicit texture LoRA path |
| `DRY_RUN=1` | Validate command setup without generating images |

## DressCode / VITON-HD Benchmarks

Set `datasets.dresscode_root` or `datasets.viton_root` in `system.json`, then run:

```bash
python inference_benchmark_testset.py --dataset dresscode
python inference_benchmark_testset.py --dataset viton
```

## VLM Scoring

After generating FittingEffect3K predictions, run:

```bash
python vlm_fit_eval.py
```

See [`evaluation.md`](evaluation.md) for the VLM protocol, API configuration, and output files.
