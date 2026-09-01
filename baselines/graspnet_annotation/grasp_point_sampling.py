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
        # Select a deterministic subset without privileging the voxelization
        # order (which is an implementation detail of the sampler).
        indices = np.sort(np.random.default_rng(int(config.seed)).choice(len(reduced), cap, replace=False))
        reduced = reduced[indices]
    return np.asarray(reduced, dtype=np.float32)


def sample_collision_points(mesh: trimesh.Trimesh, config: DenseAnnotationConfig) -> np.ndarray:
    """Build the independent 3 mm collision/width cloud (no grasp-point cap)."""

    sampled = _sample_surface(mesh, int(config.surface_samples), config.seed + 1)
    return _voxel_reduce(sampled, config.collision_voxel_size_m).astype(np.float32)


def _validate_surface_points(points_m: np.ndarray) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"surface points must have shape (N, 3), got {points.shape}")
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("surface points must be non-empty and finite")
    return points


def _sample_surface_points(points_m: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = _validate_surface_points(points_m)
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    if len(points) <= count:
        return points.copy()
    indices = np.sort(np.random.default_rng(int(seed)).choice(len(points), count, replace=False))
    return points[indices]


def sample_grasp_points_from_surface_points(points_m: np.ndarray, config: DenseAnnotationConfig,
                                            *, max_points: int | None = None) -> np.ndarray:
    """Build grasp points directly from a PLY surface cloud (no mesh reconstruction)."""
    sampled = _sample_surface_points(points_m, int(config.surface_samples), config.seed)
    reduced = _voxel_reduce(sampled, config.voxel_size_m)
    cap = int(config.max_grasp_points if max_points is None else max_points)
    if cap <= 0:
        raise ValueError("max_points must be positive")
    if len(reduced) > cap:
        indices = np.sort(np.random.default_rng(int(config.seed)).choice(len(reduced), cap, replace=False))
        reduced = reduced[indices]
    return np.asarray(reduced, dtype=np.float32)


def sample_collision_points_from_surface_points(points_m: np.ndarray, config: DenseAnnotationConfig) -> np.ndarray:
    """Build the 3 mm collision/width cloud directly from the same PLY points."""
    sampled = _sample_surface_points(points_m, int(config.surface_samples), config.seed + 1)
    return _voxel_reduce(sampled, config.collision_voxel_size_m).astype(np.float32)
