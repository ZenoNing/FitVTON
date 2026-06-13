# Configuration, Datasets, and Checkpoints

FitVTON reads paths from `system.json` through `system_config.py`. Empty fields fall back to built-in defaults until you set them.

## Path Rules

- Relative paths are resolved from the `FitVTON/` directory.
- Absolute paths are used as-is.
- `~` is expanded.

Example with repo-local assets:

```json
{
  "datasets": {
    "garmentcode_root": "GarmentCodeVTON",
    "fittingeffect_root": "FittingEffectDataset",
    "dresscode_root": "../DressCodeDataset",
    "viton_root": "../VITONDataset"
  },
  "checkpoints": {
    "root": "checkpoints"
  }
}
```

Example with external storage:

```json
{
  "datasets": {
    "garmentcode_root": "/data/vton/GarmentCodeVTON",
    "fittingeffect_root": "/data/vton/FittingEffectDataset",
    "dresscode_root": "/data/DressCodeDataset",
    "viton_root": "/data/VITONDataset"
  },
  "checkpoints": {
    "root": "/data/vton/checkpoints"
  }
}
```

Inspect resolved paths before long runs:

```bash
python system_config.py checkpoints.root
python system_config.py datasets.garmentcode_root
python system_config.py datasets.fittingeffect_root
```

## Required Layouts

The folder names on disk can differ. Only the internal structure matters.

| Key | Must contain |
|-----|--------------|
| `checkpoints.root` | `default_lora_weights.safetensors`, `texture_lora_weights.safetensors` |
| `datasets.garmentcode_root` | `Ref/`, `female/`, `male/` trees |
| `datasets.fittingeffect_root` | `tryon_triples_all.csv`, `female/`, `male/` |
| `datasets.dresscode_root` | DressCode dataset root |
| `datasets.viton_root` | VITON-HD dataset root |

## Download Assets

Install the Hugging Face CLI:

```bash
pip install -U huggingface_hub
```

Download into the repo:

```bash
hf download ZenoNing/FitVTON checkpoints --local-dir .
hf download ZenoNing/GarmentCodeVTONDataset \
  --repo-type dataset \
  --local-dir GarmentCodeVTON
hf download ZenoNing/FittingEffectDataset \
  --repo-type dataset \
  --local-dir FittingEffectDataset
```

Or download to custom locations:

```bash
hf download ZenoNing/FitVTON checkpoints --local-dir /data/vton/checkpoints
hf download ZenoNing/GarmentCodeVTONDataset \
  --repo-type dataset \
  --local-dir /data/vton/GarmentCodeVTON
hf download ZenoNing/FittingEffectDataset \
  --repo-type dataset \
  --local-dir /data/vton/FittingEffectDataset
```

Then set `system.json` to those directories.

## Checkpoints

Weights are not shipped in git. The checkpoints root should contain:

| File | Stage | Description |
|------|-------|-------------|
| `default_lora_weights.safetensors` | I | Fitting LoRA: geometry / fit prior |
| `texture_lora_weights.safetensors` | II | Texture LoRA: real-image rectification |

For inference and FittingEffect3K evaluation, these weights are enough. Retraining is optional.

## Recommended Values

| Key | Recommended value | Used for |
|-----|-------------------|----------|
| `datasets.garmentcode_root` | `GarmentCodeVTON` | Stage I simulation triplets |
| `datasets.dresscode_root` | `../DressCodeDataset` | Stage II rectification + benchmarks |
| `datasets.viton_root` | `../VITONDataset` | Benchmarks |
| `datasets.fittingeffect_root` | `FittingEffectDataset` | FittingEffect3K evaluation |
| `checkpoints.root` | `checkpoints` | Pretrained LoRA weights |
| `outputs.pseudo_images` | `pseudo_images` | Stage II pseudo targets |
| `outputs.fitting_lora` | `outputs/fitting_lora` | Stage I training output |
| `outputs.texture_lora` | `outputs/texture_lora` | Stage II training output |
| `outputs.maskhead` | `outputs/maskhead` | Mask-head training output |
| `outputs.demo` | `outputs/demo` | Demo inference output |
| `outputs.fittingeffect` | `outputs/fittingeffect` | FittingEffect3K inference output |
| `vlm_eval.gateway` | user-provided | OpenAI-compatible API base URL |
| `vlm_eval.api_key` | user-provided | API key for VLM scoring |
| `vlm_eval.model` | `gpt-5.2-chat` | VLM scoring agent |
| `vlm_eval.output_dir` | `eval_vlm` | VLM evaluation output |
| `vlm_eval.concurrency` | `8` | Concurrent API requests |

Model IDs default to Hugging Face repos `black-forest-labs/FLUX.1-Kontext-dev` and `zer0int/LongCLIP-KO-LITE-TypoAttack-Attn-ViT-L-14`. Optional local overrides: `models.flux_model_path`, `models.longclip_model_path`.

## Script Overrides

`scripts/run_fittingeffect_eval.sh` reads `system.json` by default, but these environment variables can override paths:

| Variable | Overrides |
|----------|-----------|
| `FITTINGEFFECT_ROOT` | `datasets.fittingeffect_root` |
| `CHECKPOINTS_ROOT` | `checkpoints.root` |
| `OUTPUT_DIR` | `outputs.fittingeffect` |
| `TRIPLES_CSV` | CSV path |

Most Python entry points also accept CLI path flags, e.g. `--fitting_lora_dir`, `--garmentcode_root`, and `--predict-folder`.
