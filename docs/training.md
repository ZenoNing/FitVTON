# Training

Training follows the paper's two-stage strategy:

1. Stage I trains the fitting LoRA on GarmentCodeVTON simulation triplets.
2. Stage II rectifies texture realism with pseudo-triplets from DressCode / VITON-HD while keeping the fitting LoRA frozen.

Activate the main environment first:

```bash
conda activate flux
```

## Stage I: Fitting LoRA

Stage I trains the transformer LoRA through a six-stage curriculum:

- shape-heavy dress pairs
- cloth-type-balanced full data
- wearing-style-focused two-piece data
- shape-balanced full data
- cloth-type-balanced full data
- wearing-style-focused data with one-piece retained

Run:

```bash
bash scripts/train_stage1.sh
```

Mask heads are trained automatically first when their weights are missing, then used for dual-branch supervision during LoRA training.

Useful environment overrides:

| Variable | Meaning |
|----------|---------|
| `NUM_GPUS` | Number of GPUs |
| `TRAIN_BATCH_SIZE` | Per-process training batch size |
| `CAP` | Max pairs per group |
| `SKIP_MASK_HEAD=1` | Skip mask-head pretraining |

## Stage I Smoke Test

For a quick trainer sanity check:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 train_fitting_lora.py \
  --garmentcode_root /path/to/GarmentCodeVTON \
  --output_dir outputs/fitting_lora_smoke \
  --max_train_steps 2
```

## Stage II: Texture Rectification

Generate pseudo targets with the frozen Stage I model. Pairs are listed in `pseudo.txt`.

```bash
bash scripts/generate_pseudo_images.sh
```

Train the texture LoRA on pseudo-triplets mixed with real reconstruction pairs from `second.txt`, keeping the fitting LoRA frozen:

```bash
bash scripts/train_stage2.sh
```

Stage II needs DressCode and/or VITON-HD on disk. Set `datasets.dresscode_root` and `datasets.viton_root` in `system.json`.

## Outputs

Default output locations are configured in `system.json`:

| Key | Default |
|-----|---------|
| `outputs.maskhead` | `outputs/maskhead` |
| `outputs.fitting_lora` | `outputs/fitting_lora` |
| `outputs.pseudo_images` | `pseudo_images` |
| `outputs.texture_lora` | `outputs/texture_lora` |

See [`configuration.md`](configuration.md) for path rules and external storage examples.
