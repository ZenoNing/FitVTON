#!/usr/bin/env python3
"""
Batch VLM fit evaluation for virtual try-on results (Fit-Oriented Evaluation Protocol), plus score aggregation.

Triple CSV columns: source_person, cloth, target_person
  - Prediction image: {source_person}_{cloth}.png
  - GT image: {target_person}.jpg

Category rules:
  - cloth starts with upper_   -> upper
  - cloth starts with lower_   -> lower
  - cloth starts with wholebody_ or dress_ -> dress

API credentials are read from system.json (vlm_eval.gateway, vlm_eval.api_key)
via system_config.py. Environment overrides: VLM_EVAL_GATEWAY, VLM_EVAL_API_KEY,
VLM_EVAL_MODEL.

Overall score uses macro averaging: mean of per-category averages (upper, lower,
dress), giving equal weight to each garment type.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

from system_config import (
    cfg_path,
    fittingeffect_subpath,
    vlm_eval_api_key,
    vlm_eval_concurrency,
    vlm_eval_gateway,
    vlm_eval_model,
    vlm_eval_output_dir,
)

if TYPE_CHECKING:
    from openai import OpenAI  # type: ignore[import-not-found]

CATEGORY_ORDER = ("upper", "lower", "dress")

SYSTEM_PROMPT = """You are a professional garment fitting evaluator with expertise in apparel design and human body fitting.

Your task is to compare garment fitting quality between two images of the same garment worn by the same person or people with the same body shape.

One image shows the real person wearing the garment (ground truth).
The other image shows a virtual try-on result.

You must treat the real-wearing image as the ground truth reference.

Do NOT judge visual aesthetics, image realism, fashion style, or attractiveness.
Focus strictly on fitting accuracy and consistency.

Ignore differences caused primarily by lighting, exposure, white balance, shadows, background, camera perspective, camera distance, focal length, or the person's apparent size in the image.
Do not penalize Image B just because the person appears larger, smaller, closer, farther, or framed differently.
Instead, focus on where the garment sits on the body and how it relates to specific body regions, such as shoulders, neckline, chest, waist, hips, crotch, sleeves, inseam, and hem.
Prioritize garment-to-body placement, proportion relative to body parts, silhouette around the body, and local fit behavior over global image composition.

You must follow these output rules strictly:

Scoring rubric (use INTEGER scores 1–5 only):
- 5: Virtually identical fit to Image A; only negligible differences.
- 4: Very close; minor, localized differences that do not change the perceived fit behavior.
- 3: Acceptable but clearly different; multiple noticeable differences in fit behavior/silhouette/alignment.
- 2: Poor match; major discrepancies that change key fit behavior or proportions.
- 1: Very poor; the fit behavior is largely incorrect or inconsistent with Image A.

For each required dimension:
- Provide `score` (1–5 integer) and a brief `explanation` grounded in visible differences.

Finally:
- Provide a short `summary` of the main fitting discrepancies (category-relevant).

Do NOT provide any overall fitting consistency score.
Do NOT include fields like "overall_fitting_consistency" or "overall_fitting_consistency_score".

Output ONLY a valid JSON object. Do NOT include any extra text, markdown, or code fences.

"""

USER_PROMPT_UPPER = """Please evaluate how well the virtual try-on result reproduces the garment fitting behavior observed in the real-wearing image, focusing on an upper-body garment (top).

Compare Image B against Image A along the following fitting dimensions, with emphasis on upper-body fitting:

Important rule for layered clothing:
- If the swapped upper garment is an outerwear piece or outer layer (for example, a jacket, coat, cardigan, hoodie, overshirt, or any visibly worn-over-an-inner-top garment), evaluate ONLY the fit of that outer garment.
- In such cases, ignore differences in the inner layer / undershirt / base top between Image A and Image B.
- Do not penalize Image B for mismatches in the visible inner top when the target garment being evaluated is the outerwear layer.

1. Garment–Body Alignment
2. Tightness / Looseness Consistency
3. Upper-Body Silhouette Consistency
4. Local Fit Artifacts (Upper Body)

Output JSON must use EXACTLY these keys for the 4 dimensions:
- garment_body_alignment
- tightness_looseness_consistency
- upper_body_silhouette_consistency
- local_fit_artifacts_upper_body
"""

USER_PROMPT_LOWER = """Please evaluate how well the virtual try-on result reproduces the garment fitting behavior observed in the real-wearing image, focusing on a lower-body garment (pants).

Compare Image B against Image A along the following fitting dimensions, with emphasis on lower-body fitting:

1. Garment–Body Alignment
2. Tightness / Looseness Consistency
3. Lower-Body Silhouette Consistency
4. Local Fit Artifacts (Lower Body)

Output JSON must use EXACTLY these keys for the 4 dimensions:
- garment_body_alignment
- tightness_looseness_consistency
- lower_body_silhouette_consistency
- local_fit_artifacts_lower_body
"""

USER_PROMPT_DRESS = """Please evaluate how well the virtual try-on result reproduces the garment fitting behavior observed in the real-wearing image, focusing on a dress or skirt-based garment.

Compare Image B against Image A along the following fitting dimensions, with emphasis on full-body continuity:

1. Garment–Body Alignment
2. Tightness / Looseness Consistency
3. Overall Silhouette Consistency
4. Local Fit Artifacts (Dress / Skirt)

Output JSON must use EXACTLY these keys for the 4 dimensions:
- garment_body_alignment
- tightness_looseness_consistency
- overall_silhouette_consistency
- local_fit_artifacts_dress_skirt
"""

CATEGORY_DIMENSIONS = {
    "upper": [
        "garment_body_alignment",
        "tightness_looseness_consistency",
        "upper_body_silhouette_consistency",
        "local_fit_artifacts_upper_body",
    ],
    "lower": [
        "garment_body_alignment",
        "tightness_looseness_consistency",
        "lower_body_silhouette_consistency",
        "local_fit_artifacts_lower_body",
    ],
    "dress": [
        "garment_body_alignment",
        "tightness_looseness_consistency",
        "overall_silhouette_consistency",
        "local_fit_artifacts_dress_skirt",
    ],
}

DIMENSION_DISPLAY_NAMES = {
    "garment_body_alignment": "Garment-Body Alignment",
    "tightness_looseness_consistency": "Tightness/Looseness Consistency",
    "upper_body_silhouette_consistency": "Upper Body Silhouette Consistency",
    "lower_body_silhouette_consistency": "Lower Body Silhouette Consistency",
    "overall_silhouette_consistency": "Overall Silhouette Consistency",
    "local_fit_artifacts_upper_body": "Local Fit Artifacts (Upper Body)",
    "local_fit_artifacts_lower_body": "Local Fit Artifacts (Lower Body)",
    "local_fit_artifacts_dress_skirt": "Local Fit Artifacts (Dress/Skirt)",
    "overall_fitting_consistency": "Overall Fitting Consistency",
}


def _get_gpt_client() -> "OpenAI":
    api_key = vlm_eval_api_key()
    base_url = vlm_eval_gateway()
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: openai. Install with: pip install openai"
        ) from e
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120.0,
        max_retries=2,
    )


def _image_to_data_uri(image_path: str) -> str:
    with open(image_path, "rb") as f:
        contents = f.read()
    ext = os.path.splitext(image_path)[1].lower()
    content_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    content_type = content_type_map.get(ext, "image/jpeg")
    base64_data = base64.b64encode(contents).decode("utf-8")
    return f"data:{content_type};base64,{base64_data}"


def _write_results_atomic(output_path: Path, results: List[Dict]) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)


def _strip_known_suffix(name: str) -> str:
    return re.sub(r"\.(png|jpg|jpeg|webp)$", "", name, flags=re.IGNORECASE)


def _task_key_from_names(source_person: str, cloth: str, target_person: str) -> Tuple[str, str, str]:
    return (
        _strip_known_suffix(source_person),
        _strip_known_suffix(cloth),
        _strip_known_suffix(target_person),
    )


def _task_key_from_pair(pair: Dict) -> Tuple[str, str, str]:
    return _task_key_from_names(pair["source_person"], pair["cloth"], pair["target_person"])


def _task_key_from_result(result: Dict) -> Optional[Tuple[str, str, str]]:
    source_person = result.get("source_person")
    cloth = result.get("cloth")
    target_person = result.get("target_person")
    if not source_person or not cloth or not target_person:
        return None
    return _task_key_from_names(source_person, cloth, target_person)


def _load_existing_results(output_path: Path) -> Tuple[List[Dict], Dict[Tuple[str, str, str], Dict]]:
    if not output_path.exists():
        return [], {}
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Results file is not a JSON array: {output_path}")
    successes: Dict[Tuple[str, str, str], Dict] = {}
    for item in data:
        if not isinstance(item, dict) or not item.get("ok"):
            continue
        key = _task_key_from_result(item)
        if key is not None:
            successes[key] = item
    return data, successes


def _infer_category_from_cloth(cloth_name: str) -> Tuple[Optional[str], Optional[str]]:
    cloth_name = _strip_known_suffix(cloth_name)
    if cloth_name.startswith("upper_"):
        return "upper", None
    if cloth_name.startswith("lower_"):
        return "lower", None
    if cloth_name.startswith("wholebody_") or cloth_name.startswith("dress_"):
        return "dress", None
    return None, f"Cannot infer category from cloth: {cloth_name}"


def _infer_gender_from_person(person_name: str) -> Tuple[Optional[str], Optional[str]]:
    person_name = _strip_known_suffix(person_name)
    if person_name.startswith("female_"):
        return "female", None
    if person_name.startswith("male_"):
        return "male", None
    return None, f"Cannot infer gender from person: {person_name}"


def _build_pair_from_triple_row(
    row: Dict[str, str],
    predict_folder: Path,
    gt_folder: Optional[Path],
    female_gt_folder: Optional[Path] = None,
    male_gt_folder: Optional[Path] = None,
) -> Tuple[Optional[Dict], Optional[str]]:
    source_person = _strip_known_suffix((row.get("source_person") or "").strip())
    cloth = _strip_known_suffix((row.get("cloth") or "").strip())
    target_person = _strip_known_suffix((row.get("target_person") or "").strip())

    if not source_person or not cloth or not target_person:
        return None, f"CSV row missing fields: {row}"

    category, error = _infer_category_from_cloth(cloth)
    if error:
        return None, error

    source_gender, error = _infer_gender_from_person(source_person)
    if error:
        return None, error
    target_gender, error = _infer_gender_from_person(target_person)
    if error:
        return None, error
    if source_gender != target_gender:
        return None, f"source/target gender mismatch: {source_person} vs {target_person}"

    if source_gender == "female" and female_gt_folder is not None:
        selected_gt_folder = female_gt_folder
    elif source_gender == "male" and male_gt_folder is not None:
        selected_gt_folder = male_gt_folder
    elif gt_folder is not None:
        selected_gt_folder = gt_folder
    else:
        return None, f"No GT folder configured for gender={source_gender}"

    predict_filename = f"{source_person}_{cloth}.png"
    gt_filename = f"{target_person}.jpg"

    return {
        "gt_path": str(selected_gt_folder / gt_filename),
        "predict_path": str(predict_folder / predict_filename),
        "category": category,
        "gender": source_gender,
        "predict_filename": predict_filename,
        "gt_filename": gt_filename,
        "source_person": source_person,
        "cloth": cloth,
        "target_person": target_person,
    }, None


def _load_pairs_from_csv(
    triples_csv_path: Path,
    predict_folder: Path,
    gt_folder: Optional[Path],
    female_gt_folder: Optional[Path] = None,
    male_gt_folder: Optional[Path] = None,
) -> List[Dict]:
    if not triples_csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {triples_csv_path}")

    pairs: List[Dict] = []
    with open(triples_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_fields = {"source_person", "cloth", "target_person"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"CSV missing columns: {sorted(missing_fields)}")

        for row_idx, row in enumerate(reader, start=2):
            pair, error = _build_pair_from_triple_row(
                row,
                predict_folder,
                gt_folder,
                female_gt_folder=female_gt_folder,
                male_gt_folder=male_gt_folder,
            )
            if error:
                print(f"Warning: skip CSV row {row_idx} - {error}")
                continue
            pairs.append(pair)

    return pairs


def _parse_evaluation_json(evaluation_text: str) -> Optional[Dict]:
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", evaluation_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"\{.*\}", evaluation_text, re.DOTALL)
        if not json_match:
            return None
        json_str = json_match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


async def evaluate_single_pair(
    gt_path: str,
    predict_path: str,
    category: str,
    model: str,
) -> Dict:
    user_prompt_map = {
        "upper": USER_PROMPT_UPPER,
        "lower": USER_PROMPT_LOWER,
        "dress": USER_PROMPT_DRESS,
    }
    user_prompt = user_prompt_map[category]
    gt_url = _image_to_data_uri(gt_path)
    predict_url = _image_to_data_uri(predict_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image A (Ground Truth - Real person wearing the garment):"},
                {"type": "image_url", "image_url": {"url": gt_url}},
                {"type": "text", "text": "\n\nImage B (Virtual Try-on Result):"},
                {"type": "image_url", "image_url": {"url": predict_url}},
                {"type": "text", "text": f"\n\n{user_prompt}"},
            ],
        },
    ]

    try:
        client = _get_gpt_client()
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=2000,
            seed=42,
        )
        evaluation_text = response.choices[0].message.content or ""
        evaluation_json = _parse_evaluation_json(evaluation_text)
        return {
            "ok": True,
            "gt_path": gt_path,
            "predict_path": predict_path,
            "category": category,
            "model": model,
            "evaluation": evaluation_json,
            "usage": {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            },
        }
    except Exception as e:
        return {
            "ok": False,
            "gt_path": gt_path,
            "predict_path": predict_path,
            "category": category,
            "error": str(e),
        }


async def _evaluate_pair_with_metadata(pair: Dict, index: int, total: int, model: str) -> Dict:
    print(
        f"\n[{index}/{total}] Evaluating "
        f"{pair['predict_filename']} vs {pair['gt_filename']} "
        f"(category: {pair['category']})"
    )
    result = await evaluate_single_pair(
        pair["gt_path"],
        pair["predict_path"],
        pair["category"],
        model=model,
    )
    result["predict_filename"] = pair["predict_filename"]
    result["gt_filename"] = pair["gt_filename"]
    result["source_person"] = pair["source_person"]
    result["cloth"] = pair["cloth"]
    result["target_person"] = pair["target_person"]
    result["task_index"] = index
    print(f"[{index}/{total}] Done: {'ok' if result.get('ok') else 'failed'}")
    if not result.get("ok"):
        print(f"[{index}/{total}] Error: {result.get('error', 'unknown')}")
    return result


def _eval_output_path(output_dir: Path, predict_folder: Path, triples_csv_path: Path, model: str) -> Path:
    model_tag = model.replace("/", "_").replace(":", "_")
    filename = f"vlm_eval_results_{predict_folder.name}_{triples_csv_path.stem}_{model_tag}.json"
    return output_dir / filename


async def run_batch_evaluation(
    *,
    predict_folder: str,
    triples_csv: str,
    output_dir: str,
    female_gt_folder: str,
    male_gt_folder: str,
    gt_folder: str,
    concurrency: int,
    model: str,
) -> Optional[Path]:
    predict_folder_path = Path(predict_folder)
    triples_csv_path = Path(triples_csv)
    output_dir_path = Path(output_dir)
    gt_folder_path = Path(gt_folder) if gt_folder else None
    female_gt_folder_path = Path(female_gt_folder) if female_gt_folder else None
    male_gt_folder_path = Path(male_gt_folder) if male_gt_folder else None

    if not predict_folder_path.exists():
        print(f"Error: predict folder not found: {predict_folder}")
        return None
    if gt_folder_path is None:
        if female_gt_folder_path is None or not female_gt_folder_path.exists():
            print(f"Error: female GT folder not found: {female_gt_folder}")
            return None
        if male_gt_folder_path is None or not male_gt_folder_path.exists():
            print(f"Error: male GT folder not found: {male_gt_folder}")
            return None
    if not triples_csv_path.exists():
        print(f"Error: CSV not found: {triples_csv}")
        return None

    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = _eval_output_path(output_dir_path, predict_folder_path, triples_csv_path, model)
    print(f"Writing results to: {output_path}")

    existing_results, existing_successes = [], {}
    try:
        existing_results, existing_successes = _load_existing_results(output_path)
    except Exception as e:
        print(f"Error loading existing results: {e}")
        return None

    if existing_results:
        fail_count = sum(1 for r in existing_results if isinstance(r, dict) and not r.get("ok"))
        print(
            f"Resumed results: total={len(existing_results)}, "
            f"success={len(existing_successes)}, fail={fail_count}"
        )

    try:
        pairs = _load_pairs_from_csv(
            triples_csv_path,
            predict_folder_path,
            gt_folder_path,
            female_gt_folder=female_gt_folder_path,
            male_gt_folder=male_gt_folder_path,
        )
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    print(f"Loaded {len(pairs)} pairs from CSV")

    filtered_pairs: List[Dict] = []
    for p in pairs:
        if not Path(p["gt_path"]).exists():
            print(f"Warning: GT missing, skip: {p['gt_path']}")
            continue
        if not Path(p["predict_path"]).exists():
            print(f"Warning: prediction missing, skip: {p['predict_path']}")
            continue
        filtered_pairs.append(p)
    pairs = filtered_pairs
    print(f"Pairs ready for scoring: {len(pairs)}")

    pair_order = {_task_key_from_pair(p): i for i, p in enumerate(pairs)}
    pending_pairs = [p for p in pairs if _task_key_from_pair(p) not in existing_successes]
    if len(pairs) - len(pending_pairs):
        print(f"Skipping already-scored pairs: {len(pairs) - len(pending_pairs)}")
    pairs = pending_pairs
    print(f"Pairs to submit: {len(pairs)}")

    results = list(existing_successes.values())
    total = len(pairs)
    if total == 0:
        ordered_results = sorted(
            results,
            key=lambda x: pair_order.get(_task_key_from_result(x) or ("", "", ""), 10**9),
        )
        for item in ordered_results:
            item.pop("task_index", None)
        _write_results_atomic(output_path, ordered_results)
        print("No new pairs to score; existing results saved.")
        return output_path

    concurrency = max(1, min(concurrency, total))
    print(f"Concurrency: {concurrency}")
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_one(pair: Dict, index: int) -> Dict:
        async with semaphore:
            return await _evaluate_pair_with_metadata(pair, index, total, model=model)

    tasks = [asyncio.create_task(_run_one(pair, i)) for i, pair in enumerate(pairs, 1)]
    for finished in asyncio.as_completed(tasks):
        result = await finished
        results.append(result)
        _write_results_atomic(output_path, results)

    results.sort(key=lambda x: pair_order.get(_task_key_from_result(x) or ("", "", ""), 10**9))
    for item in results:
        item.pop("task_index", None)
    _write_results_atomic(output_path, results)

    success_count = sum(1 for r in results if r.get("ok"))
    print(f"\nEvaluation finished: {success_count}/{len(results)} successful")
    print(f"Results saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def normalize_dimension_name(dim_name: str, category: str) -> Optional[str]:
    if not dim_name:
        return None

    category = (category or "").strip().lower()
    raw = str(dim_name).strip()
    raw_lower = raw.lower()

    if "summary" in raw_lower or "discrepanc" in raw_lower:
        return None
    if raw_lower.startswith("main_") and raw_lower.endswith("_discrepancies"):
        return None
    if raw_lower in {"fitting_comparison", "fitting_dimensions", "evaluation", "usage"}:
        return None

    raw = re.sub(r"^\s*\d+\s*[_\.)]\s*", "", raw)

    if re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", raw):
        if raw == "local_fit_artifacts":
            if category == "upper":
                return "local_fit_artifacts_upper_body"
            if category == "lower":
                return "local_fit_artifacts_lower_body"
            if category == "dress":
                return "local_fit_artifacts_dress_skirt"
        return raw

    key = raw_lower.replace("–", "-").replace("—", "-")
    key = re.sub(r"\s*/\s*", "/", key)
    key = re.sub(r"[()]+", "", key)
    key = re.sub(r"\s+", " ", key).strip()

    mapping = {
        "garment-body alignment": "garment_body_alignment",
        "garment body alignment": "garment_body_alignment",
        "tightness/looseness consistency": "tightness_looseness_consistency",
        "tightness looseness consistency": "tightness_looseness_consistency",
        "upper body silhouette consistency": "upper_body_silhouette_consistency",
        "lower body silhouette consistency": "lower_body_silhouette_consistency",
        "lower-body silhouette consistency": "lower_body_silhouette_consistency",
        "overall silhouette consistency": "overall_silhouette_consistency",
        "local fit artifacts upper body": "local_fit_artifacts_upper_body",
        "local fit artifacts lower body": "local_fit_artifacts_lower_body",
        "local fit artifacts dress/skirt": "local_fit_artifacts_dress_skirt",
        "local fit artifacts dress skirt": "local_fit_artifacts_dress_skirt",
        "local fit artifacts": "local_fit_artifacts",
        "overall fitting consistency": "overall_fitting_consistency",
    }

    mapped = mapping.get(key)
    if mapped == "local_fit_artifacts":
        if category == "upper":
            return "local_fit_artifacts_upper_body"
        if category == "lower":
            return "local_fit_artifacts_lower_body"
        if category == "dress":
            return "local_fit_artifacts_dress_skirt"
        return None
    return mapped


def iter_dimension_scores(evaluation: Dict) -> Iterable[Tuple[str, float]]:
    if not evaluation or not isinstance(evaluation, dict):
        return

    def _emit_from_dict(d: Dict) -> Iterable[Tuple[str, float]]:
        for k, v in (d or {}).items():
            if isinstance(v, dict) and "score" in v:
                yield k, v["score"]
            elif isinstance(v, (int, float)) and isinstance(k, str) and k.endswith("_score"):
                yield k[:-6], v

    for container_key in ("fitting_comparison", "fitting_dimensions"):
        if container_key in evaluation and isinstance(evaluation.get(container_key), dict):
            yield from _emit_from_dict(evaluation[container_key])
    yield from _emit_from_dict(evaluation)


def extract_scores(data: List[Dict]) -> Dict:
    results = defaultdict(lambda: defaultdict(list))
    for item in data:
        if not item.get("ok", False):
            continue
        category = item.get("category")
        if not category:
            continue
        evaluation = item.get("evaluation", {})
        if not evaluation:
            continue
        for dim_name, score in iter_dimension_scores(evaluation):
            clean_name = normalize_dimension_name(dim_name, category)
            if not clean_name:
                continue
            results[category][clean_name].append(score)
    return results


def calculate_averages(results: Dict) -> Dict:
    """
    Compute per-dimension and per-category averages.

    overall_avg is the macro average across garment categories: the unweighted mean
    of upper, lower, and dress category averages (only categories with scores).
    """
    averages: Dict = {}
    micro_scores: List[float] = []

    for category, dimensions in results.items():
        category_scores: List[float] = []
        dim_averages = {}

        for dim_name, scores in dimensions.items():
            if not scores:
                continue
            avg_score = sum(scores) / len(scores)
            dim_averages[dim_name] = {
                "average": round(avg_score, 2),
                "count": len(scores),
            }
            if dim_name != "overall_fitting_consistency":
                category_scores.extend(scores)
                micro_scores.extend(scores)

        category_avg = round(sum(category_scores) / len(category_scores), 2) if category_scores else 0.0
        dim_averages_filtered = {
            k: v for k, v in dim_averages.items() if k != "overall_fitting_consistency"
        }
        averages[category] = {
            "category_avg": category_avg,
            "total_count": len(category_scores),
            "dimensions": dim_averages_filtered,
        }

    category_macro_avgs = [
        averages[c]["category_avg"]
        for c in CATEGORY_ORDER
        if c in averages and averages[c]["total_count"] > 0
    ]
    overall_macro = (
        round(sum(category_macro_avgs) / len(category_macro_avgs), 2)
        if category_macro_avgs
        else 0.0
    )
    overall_micro = round(sum(micro_scores) / len(micro_scores), 2) if micro_scores else 0.0

    return {
        "overall_avg": overall_macro,
        "overall_avg_method": "macro_category",
        "overall_micro_avg": overall_micro,
        "overall_count": len(micro_scores),
        "categories_in_macro": len(category_macro_avgs),
        **averages,
    }


def get_dimension_display_name(dim_name: str) -> str:
    return DIMENSION_DISPLAY_NAMES.get(dim_name, dim_name.replace("_", " ").title())


def print_analysis(averages: Dict) -> None:
    print("=" * 80)
    print("VLM Fit Evaluation Analysis")
    print("=" * 80)
    print()

    if "overall_avg" in averages:
        print("Overall (macro average of upper / lower / dress category means)")
        print(f"  Score: {averages['overall_avg']:.2f}")
        print(f"  Categories included: {averages.get('categories_in_macro', 0)}")
        print(f"  Total dimension scores: {averages.get('overall_count', 0)}")
        if "overall_micro_avg" in averages:
            print(f"  Micro average (all scores pooled): {averages['overall_micro_avg']:.2f}")
        print()
        print("-" * 80)
        print()

    for category in CATEGORY_ORDER:
        if category not in averages:
            continue
        data = averages[category]
        print(f"[{category.upper()}]")
        print(f"  Category mean: {data['category_avg']:.2f} (n={data['total_count']})")
        print("  Per-dimension means:")
        main_dims = CATEGORY_DIMENSIONS.get(category, [])
        for dim_name in main_dims:
            if dim_name in data["dimensions"]:
                dim_data = data["dimensions"][dim_name]
                display_name = get_dimension_display_name(dim_name)
                print(f"    - {display_name:45s}: {dim_data['average']:.2f} (n={dim_data['count']})")
        other_dims = set(data["dimensions"].keys()) - set(main_dims)
        if other_dims:
            print("  Other dimensions:")
            for dim_name in sorted(other_dims):
                dim_data = data["dimensions"][dim_name]
                display_name = get_dimension_display_name(dim_name)
                print(f"    - {display_name:45s}: {dim_data['average']:.2f} (n={dim_data['count']})")
        print()
        print("-" * 80)
        print()


def build_analysis_output_path(input_file: str) -> str:
    input_path = Path(input_file)
    stem = input_path.stem
    if stem.startswith("vlm_eval_results_"):
        stem = stem.replace("vlm_eval_results_", "vlm_eval_analysis_", 1)
    elif stem.startswith("gpt_eval_results_"):
        stem = stem.replace("gpt_eval_results_", "vlm_eval_analysis_", 1)
    else:
        stem = f"{stem}_analysis"
    return str(input_path.with_name(f"{stem}.json"))


def run_analysis(input_file: str, output_file: Optional[str] = None) -> Dict:
    output_file = output_file or build_analysis_output_path(input_file)
    print(f"Loading results from: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")

    print("Extracting scores...")
    results = extract_scores(data)
    print("Computing averages...")
    averages = calculate_averages(results)
    print()
    print_analysis(averages)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(averages, f, indent=2, ensure_ascii=False)
    print(f"Analysis saved to: {output_file}")
    return averages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit-oriented VLM evaluation and score aggregation for try-on results."
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip API evaluation; only analyze an existing results JSON.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Run evaluation only; do not compute aggregates afterward.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Results JSON for --analyze-only.",
    )
    parser.add_argument(
        "--analysis-output",
        type=str,
        default=None,
        help="Output path for analysis JSON.",
    )
    parser.add_argument(
        "--predict-folder",
        type=str,
        default=None,
        help="Directory with predicted try-on PNGs.",
    )
    parser.add_argument(
        "--gt-folder",
        type=str,
        default="",
        help="Single GT folder; overrides gender-specific folders when set.",
    )
    parser.add_argument(
        "--female-gt-folder",
        type=str,
        default=None,
        help="Female GT portraits folder.",
    )
    parser.add_argument(
        "--male-gt-folder",
        type=str,
        default=None,
        help="Male GT portraits folder.",
    )
    parser.add_argument(
        "--triples-csv",
        type=str,
        default=None,
        help="Try-on triples CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for evaluation JSON output.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Concurrent API requests (default from system.json vlm_eval.concurrency).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="VLM scoring agent (default from system.json vlm_eval.model).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.analyze_only:
        if not args.input_file:
            raise SystemExit("--analyze-only requires --input-file")
        run_analysis(args.input_file, args.analysis_output)
        return

    predict_folder = args.predict_folder or str(cfg_path("outputs", "fittingeffect"))
    female_gt_folder = args.female_gt_folder or fittingeffect_subpath("female", "human")
    male_gt_folder = args.male_gt_folder or fittingeffect_subpath("male", "human")
    triples_csv = args.triples_csv or fittingeffect_subpath("tryon_triples_all.csv")
    output_dir = args.output_dir or vlm_eval_output_dir()
    concurrency = args.concurrency if args.concurrency is not None else vlm_eval_concurrency()
    model = args.model or vlm_eval_model()

    result_path = asyncio.run(
        run_batch_evaluation(
            predict_folder=predict_folder,
            triples_csv=triples_csv,
            output_dir=output_dir,
            female_gt_folder=female_gt_folder,
            male_gt_folder=male_gt_folder,
            gt_folder=args.gt_folder,
            concurrency=concurrency,
            model=model,
        )
    )

    if result_path and not args.skip_analysis:
        run_analysis(str(result_path), args.analysis_output)


if __name__ == "__main__":
    main()
