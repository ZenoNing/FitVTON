# FittingEffect3K and VLM Fit Evaluation

## FittingEffect3K

FittingEffect3K is a real-world benchmark for fit-oriented evaluation: 3,350 triplets built from 14 medium-sized garments with measurements, 5 male and 5 female models with body measurements, and 5 poses.

Download from [ZenoNing/FittingEffectDataset](https://huggingface.co/datasets/ZenoNing/FittingEffectDataset), then set `datasets.fittingeffect_root` in `system.json`.

Expected layout:

```text
FittingEffectDataset/
├── tryon_triples_all.csv
├── female/
│   ├── human/
│   └── cloth/
└── male/
    ├── human/
    └── cloth/
```

Generate predictions:

```bash
bash scripts/run_fittingeffect_eval.sh
```

## VLM API Configuration

`vlm_fit_eval.py` implements the paper's **Fit-Oriented Evaluation Protocol**. It uses an OpenAI-compatible vision API and defaults to GPT-5.2 as the VLM scoring agent.

Install the API package:

```bash
pip install openai
```

Configure `system.json`:

```json
{
  "vlm_eval": {
    "gateway": "https://your-api.example.com/v1",
    "api_key": "sk-...",
    "model": "gpt-5.2-chat",
    "output_dir": "eval_vlm",
    "concurrency": "8"
  }
}
```

Environment overrides:

- `VLM_EVAL_GATEWAY`
- `VLM_EVAL_API_KEY`
- `VLM_EVAL_MODEL`

## Run Scoring

If `system.json` is configured:

```bash
python vlm_fit_eval.py
```

Explicit path overrides:

```bash
python vlm_fit_eval.py \
  --predict-folder /scratch/vton/predictions \
  --triples-csv /data/vton/FittingEffectDataset/tryon_triples_all.csv \
  --female-gt-folder /data/vton/FittingEffectDataset/female/human \
  --male-gt-folder /data/vton/FittingEffectDataset/male/human
```

Useful flags:

| Flag | Purpose |
|------|---------|
| `--skip-analysis` | Run API scoring only |
| `--analyze-only --input-file path/to/results.json` | Recompute aggregates from existing results |
| `--analysis-output path.json` | Custom analysis JSON path |
| `--model`, `--concurrency`, `--output-dir` | Override `system.json` defaults |

Resume behavior: if the results JSON already exists, successfully scored pairs (`ok: true`) are skipped on the next run.

## Image Pairing

Triples are read from a CSV with columns:

- `source_person`
- `cloth`
- `target_person`

Path patterns:

| Role | Path pattern |
|------|--------------|
| Prediction (Image B) | `{predict_folder}/{source_person}_{cloth}.png` |
| Ground truth (Image A) | `{female\|male}/human/{target_person}.jpg` |

Gender is inferred from the `female_` / `male_` prefix on person IDs. Source and target must share the same gender.

Garment category is inferred from `cloth`:

| Prefix | Category | Silhouette dimension |
|--------|----------|----------------------|
| `upper_` | upper | Upper-Body Silhouette Consistency |
| `lower_` | lower | Lower-Body Silhouette Consistency |
| `wholebody_` or `dress_` | dress | Overall Silhouette Consistency |

## Scoring Protocol

For each pair, the VLM evaluator receives:

- **Image A**: real try-on reference
- **Image B**: virtual try-on result

It rates four fit dimensions on an integer 1-5 scale:

| Score | Meaning |
|-------|---------|
| 5 | Virtually identical fit; negligible differences |
| 4 | Very close; minor localized differences |
| 3 | Acceptable but clearly different |
| 2 | Poor match; major fit discrepancies |
| 1 | Very poor; largely incorrect fit behavior |

The evaluator focuses on fit accuracy rather than realism or aesthetics, and ignores lighting, background, camera distance, and framing differences.

Dimensions:

| Dimension | Abbrev. | Upper | Lower | Dress |
|-----------|---------|:-----:|:-----:|:-----:|
| Garment-Body Alignment | GB | yes | yes | yes |
| Tightness / Looseness Consistency | T/L | yes | yes | yes |
| Silhouette Consistency | SC | upper body | lower body | overall |
| Local Fit Artifacts | LF | upper body | lower body | dress/skirt |

For layered upper garments such as jackets or coats, only the outer garment fit is evaluated; inner-layer mismatches are ignored.

## Aggregation

`vlm_fit_eval.py` computes:

1. Per-dimension averages within each category.
2. Category averages for upper, lower, and dress.
3. `overall_avg`, the macro average across category averages:

```text
Whole Avg = (Upper Avg + Lower Avg + Dress Avg) / 3
```

Only categories with scored samples are included. `overall_micro_avg` is also reported for reference, but the paper's **Whole Avg** uses the macro category mean.

## Output Files

| File | Description |
|------|-------------|
| `eval_vlm/vlm_eval_results_{predict}_{csv}_{model}.json` | Per-pair raw results |
| `eval_vlm/vlm_eval_analysis_{predict}_{csv}_{model}.json` | Aggregated averages |

Example analysis:

```json
{
  "overall_avg": 3.08,
  "overall_avg_method": "macro_category",
  "overall_micro_avg": 3.05,
  "upper": { "category_avg": 3.22, "dimensions": { "...": "..." } },
  "lower": { "category_avg": 2.99, "dimensions": { "...": "..." } },
  "dress": { "category_avg": 2.90, "dimensions": { "...": "..." } }
}
```
