"""Resolve GarmentCodeV2 paths from system.json at the repository root."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pygarment.data_config import Properties

# GarmentCodeV2 repository root (parent of this package).
REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_JSON = REPO_ROOT / "system.json"

# Conventional model filenames under smplx_models_dir / smpl_models_dir (override in scripts if needed).
SMPLX_MODEL_CANDIDATES = {
    "female": ("SMPLX_FEMALE.npz", "SMPLX_FEMALE.pkl"),
    "male": ("SMPLX_MALE.npz", "SMPLX_MALE.pkl"),
    "neutral": ("SMPLX_NEUTRAL.npz", "SMPLX_NEUTRAL.pkl"),
}
SMPL_MODEL_CANDIDATES = {
    "female": ("SMPL_FEMALE.pkl",),
    "male": ("SMPL_MALE.pkl",),
    "neutral": ("SMPL_NEUTRAL.pkl",),
}


@lru_cache(maxsize=4)
def load_system_properties(system_path: str | None = None) -> Properties:
    path = Path(system_path) if system_path else SYSTEM_JSON
    if not path.is_absolute():
        path = REPO_ROOT / path
    return Properties(str(path))


def resolve(path_value: str, *, root: Path | None = None) -> Path:
    root = root or REPO_ROOT
    path = Path(path_value)
    return path if path.is_absolute() else (root / path).resolve()


def get_path(key: str, system_path: str | None = None) -> Path:
    props = load_system_properties(system_path)
    return resolve(props[key], root=REPO_ROOT)


def smplx_models_dir(system_path: str | None = None) -> Path:
    return get_path("smplx_models_dir", system_path)


def smpl_models_dir(system_path: str | None = None) -> Path:
    return get_path("smpl_models_dir", system_path)


def _first_existing(models_dir: Path, filenames: tuple[str, ...]) -> Path:
    for name in filenames:
        path = models_dir / name
        if path.exists():
            return path
    return models_dir / filenames[0]


def smplx_model_path_str(gender: str = "female", system_path: str | None = None) -> str:
    """Path to an SMPL-X model file; gender selects filename candidates under smplx_models_dir."""
    candidates = SMPLX_MODEL_CANDIDATES.get(gender, SMPLX_MODEL_CANDIDATES["female"])
    return str(_first_existing(smplx_models_dir(system_path), candidates))


def smpl_model_path_str(variant: str = "female", system_path: str | None = None) -> str:
    """Path to an SMPL model file; variant selects filename candidates under smpl_models_dir."""
    candidates = SMPL_MODEL_CANDIDATES.get(variant, SMPL_MODEL_CANDIDATES["female"])
    return str(_first_existing(smpl_models_dir(system_path), candidates))
