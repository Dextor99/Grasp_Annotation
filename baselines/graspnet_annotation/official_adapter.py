"""Lazy boundary around the official GraspNetAPI/Dex-Net implementation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np


class OfficialBackendUnavailable(RuntimeError):
    """Raised when an official evaluator cannot be imported."""


def require_official_backend() -> tuple[ModuleType, ModuleType]:
    """Import the official API and its utility module, failing closed."""

    required = ("graspnetAPI", "graspnetAPI.utils.utils")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise OfficialBackendUnavailable(
            "GN-Full requires graspnetAPI (including its Dex-Net evaluator). "
            "Install the baseline environment dependencies first. Missing: " + ", ".join(missing)
        )
    try:
        return tuple(importlib.import_module(name) for name in required)  # type: ignore[return-value]
    except ImportError as exc:
        raise OfficialBackendUnavailable("graspnetAPI imported incompletely; resolve its native dependencies") from exc


@dataclass(frozen=True)
class PointEvaluation:
    """Per-point candidate results with `(view, angle, depth)` axes."""

    widths_m: np.ndarray
    collision: np.ndarray
    mu_min: np.ndarray

    def __post_init__(self) -> None:
        widths = np.asarray(self.widths_m, dtype=np.float32)
        collision = np.asarray(self.collision, dtype=bool)
        scores = np.asarray(self.mu_min, dtype=np.float32)
        if widths.shape != collision.shape or scores.shape != collision.shape:
            raise ValueError("widths_m, collision, and mu_min must have the same shape")
        if widths.ndim != 3:
            raise ValueError("evaluation tensors must have shape (views, angles, depths)")
        if not np.isfinite(widths).all() or not np.isfinite(scores).all():
            raise ValueError("evaluation tensors must be finite")
        object.__setattr__(self, "widths_m", widths)
        object.__setattr__(self, "collision", collision)
        object.__setattr__(self, "mu_min", scores)


def load_dexnet_model(data_path: str | Path):
    """Load a paired ``<prefix>.obj``/``<prefix>.sdf`` with official code."""

    require_official_backend()
    try:
        eval_utils = importlib.import_module("graspnetAPI.utils.eval_utils")
        return eval_utils.load_dexnet_model(str(Path(data_path)))
    except (ImportError, OSError, ValueError) as exc:
        raise OfficialBackendUnavailable(f"Unable to load official OBJ/SDF pair from {data_path}") from exc
