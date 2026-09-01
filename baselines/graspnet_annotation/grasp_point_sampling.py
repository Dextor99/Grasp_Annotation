"""Deterministic surface-point sampling for the GN-Full baseline."""

from __future__ import annotations

import numpy as np
import trimesh

from .config import DenseAnnotationConfig


def _sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        points, _ = trimesh.sample.sample_surface_even(mesh, int(count))
        if len(points) < int(count):
            extra, _ = trimesh.sample.sample_surface(mesh, int(count) - len(points))
            points = np.vstack((points, extra))
    finally:
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float32)


def _voxel_reduce(points_m: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if len(points_m) == 0:
        return points_m.astype(np.float32)
    keys = np.floor(points_m / float(voxel_size_m)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points_m[np.sort(first)]


def sample_grasp_points(mesh: trimesh.Trimesh, config: DenseAnnotationConfig, *, max_points: int | None = None) -> np.ndarray:
    """Sample, voxel-reduce, and cap object points without changing the mesh."""

    sampled = _sample_surface(mesh, max(int(config.surface_samples), int(config.max_grasp_points)), config.seed)
    reduced = _voxel_reduce(sampled, config.voxel_size_m)
    cap = int(config.max_grasp_points if max_points is None else max_points)
    if cap <= 0:
        raise ValueError("max_points must be positive")
    if len(reduced) > cap:
        indices = np.linspace(0, len(reduced) - 1, cap, dtype=np.int64)
        reduced = reduced[indices]
    return np.asarray(reduced, dtype=np.float32)
