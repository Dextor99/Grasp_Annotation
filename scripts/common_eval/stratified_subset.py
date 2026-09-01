"""Pure deterministic sampling helpers for GN geometry-valid candidates."""

from __future__ import annotations

from typing import Any

import numpy as np


def _allocate_quotas(available: np.ndarray, target: int) -> np.ndarray:
    if available.ndim != 1:
        raise ValueError("available counts must be one-dimensional")
    if target <= 0:
        raise ValueError("target_count must be positive")
    target = min(int(target), int(available.sum()))
    quotas = np.minimum(available, target // len(available)).astype(np.int64)
    remaining = target - int(quotas.sum())
    while remaining:
        progress = False
        for point_id in range(len(available)):
            if remaining == 0:
                break
            if quotas[point_id] < available[point_id]:
                quotas[point_id] += 1
                remaining -= 1
                progress = True
        if not progress:
            raise RuntimeError("unable to allocate the requested sample budget")
    return quotas


def sample_stratified_ids(
    collision: np.ndarray,
    *,
    target_count: int = 10_000,
    candidates_per_point: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample valid flattened candidate ids evenly across grasp points.

    ``collision`` follows the GN layout ``(point, view, angle, depth)`` (or a
    two-dimensional equivalent).  Candidate ids use NumPy's C-order flattening
    and are therefore compatible with ``np.unravel_index`` used by the shard
    force-closure executor.  The returned ids are sorted for stable manifests;
    the RNG only determines which candidates within each point are selected.
    """
    labels = np.asarray(collision, dtype=bool)
    if labels.ndim < 2:
        raise ValueError("collision must have a grasp-point dimension and candidate dimensions")
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if labels.shape[0] == 0:
        raise ValueError("collision must contain at least one grasp point")
    point_stride = int(np.prod(labels.shape[1:], dtype=np.int64))
    if candidates_per_point is not None:
        if int(candidates_per_point) <= 0:
            raise ValueError("candidates_per_point must be positive")
        if int(candidates_per_point) != point_stride:
            raise ValueError(
                f"candidates_per_point={candidates_per_point} does not match collision shape ({point_stride})"
            )
    n_points = int(labels.shape[0])
    flat = labels.reshape(n_points, point_stride)
    valid_by_point = [np.flatnonzero(~flat[point_id]).astype(np.int64) for point_id in range(n_points)]
    available = np.asarray([len(ids) for ids in valid_by_point], dtype=np.int64)
    quotas = _allocate_quotas(available, int(target_count))
    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    for point_id, (valid_local_ids, quota) in enumerate(zip(valid_by_point, quotas)):
        if quota:
            local = rng.choice(valid_local_ids, size=int(quota), replace=False)
            selected_parts.append(np.asarray(local, dtype=np.int64) + point_id * point_stride)
    selected = np.sort(np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64))
    details: dict[str, Any] = {
        "candidate_ids": selected,
        "seed": int(seed),
        "target_size": int(target_count),
        "selected_size": int(len(selected)),
        "n_geometry_valid": int(available.sum()),
        "n_grasp_points": n_points,
        "collision_shape": [int(value) for value in labels.shape],
        "candidates_per_point": point_stride,
        "available_per_point": available.tolist(),
        "selected_per_point": quotas.tolist(),
        "strategy": "point_stratified_without_replacement_v1",
        "flatten_order": "C",
    }
    return selected, details


def build_sampling_manifest(
    details: dict[str, Any],
    *,
    source_geometry_run: str | None = None,
) -> dict[str, Any]:
    """Convert sampling details into a JSON-serialisable provenance manifest."""
    manifest = {key: value for key, value in details.items() if key != "candidate_ids"}
    if source_geometry_run is not None:
        manifest["source_geometry_run"] = str(source_geometry_run)
    return manifest
