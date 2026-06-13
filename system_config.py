"""Load FitVTON paths and model IDs from system.json."""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

FITVTON_ROOT = Path(__file__).resolve().parent
SYSTEM_JSON = FITVTON_ROOT / "system.json"

_BUILTIN_DEFAULTS: dict[str, dict[str, str]] = {
    "models": {
        "flux_kontext_model_id": "black-forest-labs/FLUX.1-Kontext-dev",
        "longclip_model_id": "zer0int/LongCLIP-KO-LITE-TypoAttack-Attn-ViT-L-14",
        "flux_model_path": "",
        "longclip_model_path": "",
    },
    "datasets": {
        "garmentcode_root": "../GarmentCodeVTON",
        "dresscode_root": "../DressCodeDataset",
        "viton_root": "../VITONDataset",
        "fittingeffect_root": "FittingEffectDataset",
    },
    "checkpoints": {
        "root": "checkpoints",
    },
    "outputs": {
        "pseudo_images": "pseudo_images",
        "fitting_lora": "outputs/fitting_lora",
        "texture_lora": "outputs/texture_lora",
        "maskhead": "outputs/maskhead",
        "demo": "outputs/demo",
        "fittingeffect": "outputs/fittingeffect",
    },
    "vlm_eval": {
        "gateway": "",
        "api_key": "",
        "model": "gpt-5.2-chat",
        "output_dir": "eval_vlm",
        "concurrency": "8",
    },
}

_PATH_SECTIONS = frozenset({"datasets", "checkpoints", "outputs", "vlm_eval"})

FITTING_LORA_WEIGHT_NAME = "default_lora_weights.safetensors"
TEXTURE_LORA_WEIGHT_NAME = "texture_lora_weights.safetensors"
PSEUDO_PAIRS_FILE = FITVTON_ROOT / "pseudo.txt"
SECOND_STAGE_PAIRS_FILE = FITVTON_ROOT / "second.txt"


def _is_unset(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        elif not _is_unset(value):
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def load_config() -> dict[str, dict[str, str]]:
    config = json.loads(json.dumps(_BUILTIN_DEFAULTS))
    if SYSTEM_JSON.exists():
        with SYSTEM_JSON.open("r", encoding="utf-8") as f:
            user_config = json.load(f)
        if isinstance(user_config, dict):
            config = _deep_merge(config, user_config)
    return config


def cfg(section: str, key: str) -> str:
    section_data = load_config().get(section, {})
    if key not in section_data:
        raise KeyError(f"Unknown config key: {section}.{key}")
    return str(section_data[key])


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (FITVTON_ROOT / path).resolve()
    return path


def cfg_path(section: str, key: str) -> str:
    return str(resolve_path(cfg(section, key)))


def fittingeffect_subpath(*parts: str) -> str:
    root = resolve_path(cfg("datasets", "fittingeffect_root"))
    return str(root.joinpath(*parts))


def checkpoints_root() -> Path:
    return resolve_path(cfg("checkpoints", "root"))


def fitting_lora_checkpoint() -> str:
    return str(checkpoints_root() / FITTING_LORA_WEIGHT_NAME)


def texture_lora_checkpoint() -> str:
    return str(checkpoints_root() / TEXTURE_LORA_WEIGHT_NAME)


def pseudo_pairs_file() -> str:
    return str(PSEUDO_PAIRS_FILE)


def second_stage_pairs_file() -> str:
    return str(SECOND_STAGE_PAIRS_FILE)


def flux_model_id() -> str:
    return cfg("models", "flux_kontext_model_id")


def longclip_model_id() -> str:
    return cfg("models", "longclip_model_id")


def _vlm_eval_value(key: str) -> str:
    env_map = {
        "gateway": "VLM_EVAL_GATEWAY",
        "api_key": "VLM_EVAL_API_KEY",
        "model": "VLM_EVAL_MODEL",
    }
    env_key = env_map.get(key)
    if env_key:
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            return env_val
    section = load_config().get("vlm_eval", {})
    if key not in section or _is_unset(section[key]):
        raise KeyError(f"vlm_eval.{key} is not set in system.json (or via env override)")
    return str(section[key]).strip()


def vlm_eval_gateway() -> str:
    return _vlm_eval_value("gateway").rstrip("/")


def vlm_eval_api_key() -> str:
    return _vlm_eval_value("api_key")


def vlm_eval_model() -> str:
    return _vlm_eval_value("model")


def vlm_eval_output_dir() -> str:
    return cfg_path("vlm_eval", "output_dir")


def vlm_eval_concurrency() -> int:
    raw = _vlm_eval_value("concurrency")
    return max(1, int(raw))


def local_model_override(repo_id: str) -> Path | None:
    config = load_config()["models"]
    if repo_id == config["flux_kontext_model_id"]:
        env_path = os.environ.get("FLUX_MODEL_PATH", "").strip()
        local_path = env_path or config.get("flux_model_path", "").strip()
        if local_path:
            candidate = resolve_path(local_path)
            return candidate if candidate.exists() else None
    if repo_id == config["longclip_model_id"]:
        env_path = os.environ.get("LONGCLIP_MODEL_PATH", "").strip()
        local_path = env_path or config.get("longclip_model_path", "").strip()
        if local_path:
            candidate = resolve_path(local_path)
            return candidate if candidate.exists() else None
    return None


def _parse_dot_key(dot_key: str) -> tuple[str, str]:
    if "." not in dot_key:
        raise ValueError(f"Expected section.key, got {dot_key!r}")
    section, key = dot_key.split(".", 1)
    return section, key


def _cli_main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python system_config.py <section.key>", file=sys.stderr)
        raise SystemExit(2)

    section, key = _parse_dot_key(sys.argv[1])
    value = cfg(section, key)
    if section in _PATH_SECTIONS:
        print(cfg_path(section, key))
    else:
        print(value)


if __name__ == "__main__":
    _cli_main()
