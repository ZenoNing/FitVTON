# GarmentCodeV2 and Local Simulation Data

The released GarmentCodeVTON dataset is available on Hugging Face. You only need this page if you want to regenerate or extend the simulation data locally.

Training and evaluation with the released datasets do **not** require SMPL or SMPL-X. They are only needed for local GarmentCodeV2 simulation.

## SMPL / SMPL-X Body Models

GarmentCodeV2 drives garments on **SMPL-X** bodies and uses **SMPL** as a vertical-alignment reference. These model files are subject to separate licenses and are not redistributed with this repo.

Register and download:

| Model | Official site | License |
|-------|---------------|---------|
| SMPL | [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) | Registration + SMPL Model License |
| SMPL-X | [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) | Registration + SMPL-X Model License |

Place files under:

```text
GarmentCodeV2/human_model_files/
├── smpl/
│   ├── SMPL_FEMALE.pkl
│   └── SMPL_MALE.pkl
└── smplx/
    ├── SMPLX_FEMALE.npz or SMPLX_FEMALE.pkl
    └── SMPLX_MALE.npz or SMPLX_MALE.pkl
```

`.npz` is preferred when both formats are available. The bundled body-part JSON mapping files can stay in these folders.

`GarmentCodeV2/system.json` ships with these defaults:

```json
{
  "smpl_models_dir": "./human_model_files/smpl",
  "smplx_models_dir": "./human_model_files/smplx"
}
```

## Prerequisites

Build Warp first. See [`setup.md`](setup.md#build-warp).

Then run GarmentCodeV2 commands from its directory:

```bash
cd GarmentCodeV2
conda activate flux
```

Required assets:

- SMPL / SMPL-X body models
- pose vectors (`.npz` files under `GarmentCodeV2/smplx/`)
- garment patterns under `GarmentCodeV2/assets/newcloth/`

## GarmentCodeV2 Configuration

Default keys in `GarmentCodeV2/system.json`:

| Key | Recommended value | Purpose |
|-----|-------------------|---------|
| `smplx_models_dir` | `./human_model_files/smplx` | SMPL-X model files |
| `smpl_models_dir` | `./human_model_files/smpl` | SMPL model files |
| `pose_dir` | `./smplx` | Pose `.npz` directory |
| `dataset_root` | `./outputs/vton_dataset` | Final GarmentCodeVTON layout |
| `demo_root` | `./outputs/vton_demo` | Frame-by-frame demo outputs |

After generation, set FitVTON's repo-root `system.json`:

```json
{
  "datasets": {
    "garmentcode_root": "GarmentCodeV2/outputs/vton_dataset"
  }
}
```

Any path is valid as long as it contains the expected GarmentCodeVTON layout.

## Pipeline Overview

| Script | Role |
|--------|------|
| `multi_cloth_ref_vton.py` | Render flat reference garments (`Ref/<unit>/render_*.png`) on a hidden neutral body |
| `multi_cloth_batch_vton.py` | Main batch simulator: outfits x wearing modes x bodies x poses x genders |
| `multi_cloth_demo_frames.py` | Optional frame-by-frame visualization |

Shared helpers live in `GarmentCodeV2/scripts/`.

## Output Layout

`multi_cloth_batch_vton.py` writes the layout expected by `dataset.py`:

```text
<dataset_root>/
├── Ref/
│   ├── upper1/render_front.png
│   ├── pants1/render_front.png
│   └── ...
├── female/
│   └── female0/
│       └── dress1/
│           └── one_piece/
│               └── pose0/
│                   ├── render_front.png
│                   ├── garment_mask_front.png
│                   └── body_mask_front.png
└── male/
    └── ...
```

Two-piece outfits produce both `tucked_in/` and `untucked/` subfolders; dresses use `one_piece/`.

## Step 1: Reference Garments

```bash
python multi_cloth_ref_vton.py --gpus 0
```

Useful flags:

- `--unit-filter upper1,pants4`
- `--limit 2`
- `--force`
- `--debug`

## Step 2: Batch Simulation

Full run:

```bash
python multi_cloth_batch_vton.py \
  --dataset-root ./outputs/vton_dataset \
  --cache-root ./precomputed \
  --gpus 0,1,2,3 \
  --workers-per-gpu 1
```

Smoke test:

```bash
python multi_cloth_batch_vton.py \
  --outfit-filter dress1 \
  --outfit-limit 1 \
  --pose-limit 1 \
  --num-bodies 1 \
  --genders female \
  --gpus 0
```

Other useful flags:

- `--precompute-only`
- `--force`
- `--alignment-reference female`

Progress is logged to `<cache-root>/batch_results.jsonl`. Intermediate caches live under `<cache-root>/garments/` and `<cache-root>/body_sequences/`.

## Step 3: Demo Frames

```bash
python multi_cloth_demo_frames.py --gpu-id 0
```

Target one triple:

```bash
python multi_cloth_demo_frames.py \
  --sample-id female__female0__dress1__one_piece__pose5 \
  --gpu-id 0
```

Outputs land under `demo_root` with per-frame PNGs in `frames/`.
