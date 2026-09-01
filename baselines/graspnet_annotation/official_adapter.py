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


def evaluate_official_collision(
    grasps: np.ndarray,
    model_points_m: np.ndarray,
    scene_points_m: np.ndarray | None = None,
    *,
    outlier_m: float = 0.05,
    empty_thresh: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the official GraspNet collision/empty test for one object.

    ``grasps`` uses the official ``(quality, width, height, depth, R9, t3,
    object_id)`` layout.  The helper deliberately does not require an SDF:
    the official collision stage only needs the model and scene point cloud;
    the paired OBJ/SDF is loaded later by the force-closure scorer.
    """

    grasps = np.asarray(grasps, dtype=np.float32)
    model = np.asarray(model_points_m, dtype=np.float32)
    scene = model if scene_points_m is None else np.asarray(scene_points_m, dtype=np.float32)
    if grasps.ndim != 2 or grasps.shape[1] != 17:
        raise ValueError(f"grasps must have shape (N, 17), got {grasps.shape}")
    if model.ndim != 2 or model.shape[1] != 3 or len(model) == 0:
        raise ValueError("model_points_m must be a non-empty (N, 3) array")
    if scene.ndim != 2 or scene.shape[1] != 3 or len(scene) == 0:
        raise ValueError("scene_points_m must be a non-empty (N, 3) array")
    try:
        eval_utils = importlib.import_module("graspnetAPI.utils.eval_utils")
    except ImportError as exc:
        raise OfficialBackendUnavailable("official collision evaluator is unavailable") from exc
    collision, empty = eval_utils.collision_detection(
        [grasps], [model], [None], [np.eye(4, dtype=np.float32)], scene,
        outlier=float(outlier_m), empty_thresh=int(empty_thresh), return_dexgrasps=False,
    )
    return np.asarray(collision[0], dtype=bool), np.asarray(empty[0], dtype=bool)
