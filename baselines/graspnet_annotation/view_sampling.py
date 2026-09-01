"""Official GraspNet viewpoint and pose-convention helpers."""

from __future__ import annotations

import importlib

import numpy as np


def _official_utils():
    try:
        return importlib.import_module("graspnetAPI.utils.utils")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("GN-Full requires graspnetAPI.utils.utils for official view sampling") from exc


def generate_views(num_views: int) -> np.ndarray:
    """Return views in the official GraspNet Fibonacci ordering."""

    if int(num_views) <= 0:
        raise ValueError("num_views must be positive")
    views = np.asarray(_official_utils().generate_views(int(num_views)), dtype=np.float32)
    if views.shape != (int(num_views), 3):
        raise ValueError(f"official generate_views returned {views.shape}, expected {(int(num_views), 3)}")
    return views


def make_offsets(config) -> np.ndarray:
    """Build raw `(angle, depth, width)` offset slots for every view."""

    angles = np.arange(config.num_angles, dtype=np.float32) * (np.pi / config.num_angles)
    depths = np.asarray(config.depths_m, dtype=np.float32)
    offsets = np.zeros((config.num_views, config.num_angles, len(depths), 3), dtype=np.float32)
    offsets[..., 0] = angles[None, :, None]
    offsets[..., 1] = depths[None, None, :]
    return offsets


def generate_view_rotations(view_directions: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Use the official convention, whose approach vector is ``-view``."""

    views = np.asarray(view_directions, dtype=np.float32)
    angle_values = np.asarray(angles, dtype=np.float32)
    if views.ndim != 2 or views.shape[1] != 3 or angle_values.shape != (len(views),):
        raise ValueError("view_directions must be (N, 3) and angles must be (N,)")
    rotations = _official_utils().batch_viewpoint_params_to_matrix(-views, angle_values)
    return np.asarray(rotations, dtype=np.float32)
