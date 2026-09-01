"""Bounded-memory expansion of the GN-Full candidate topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import DenseAnnotationConfig
from .view_sampling import generate_view_rotations, generate_views, make_offsets


@dataclass(frozen=True)
class CandidateBatch:
    point_indices: np.ndarray
    view_ids: np.ndarray
    angle_ids: np.ndarray
    depth_ids: np.ndarray
    points_m: np.ndarray
    view_directions: np.ndarray
    rotations: np.ndarray
    offsets: np.ndarray

    @property
    def size(self) -> int:
        return int(self.point_indices.size)

    @property
    def point_index(self) -> int:
        if self.size == 0 or not np.all(self.point_indices == self.point_indices[0]):
            raise ValueError("point_index is only defined for a single-point batch")
        return int(self.point_indices[0])


def _single_point_indices(config: DenseAnnotationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    view_ids, angle_ids, depth_ids = np.meshgrid(
        np.arange(config.num_views, dtype=np.int16),
        np.arange(config.num_angles, dtype=np.int8),
        np.arange(len(config.depths_m), dtype=np.int8),
        indexing="ij",
    )
    return view_ids.reshape(-1), angle_ids.reshape(-1), depth_ids.reshape(-1)


def iter_candidate_batches(points_m: np.ndarray, config: DenseAnnotationConfig, point_batch_size: int = 1) -> Iterator[CandidateBatch]:
    """Yield candidates in official `(point, view, angle, depth)` order."""

    points = np.asarray(points_m, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_m must have shape (N, 3), got {points.shape}")
    if int(point_batch_size) <= 0:
        raise ValueError("point_batch_size must be positive")

    views = generate_views(config.num_views)
    offsets_grid = make_offsets(config)
    view_one, angle_one, depth_one = _single_point_indices(config)
    angle_values = angle_one.astype(np.float32) * (np.pi / config.num_angles)
    per_point = config.candidates_per_point
    for start in range(0, len(points), int(point_batch_size)):
        stop = min(start + int(point_batch_size), len(points))
        count = stop - start
        point_indices = np.repeat(np.arange(start, stop, dtype=np.int32), per_point)
        view_ids = np.tile(view_one, count)
        angle_ids = np.tile(angle_one, count)
        depth_ids = np.tile(depth_one, count)
        view_directions = views[view_ids]
        rotations = generate_view_rotations(view_directions, np.tile(angle_values, count))
        yield CandidateBatch(
            point_indices=point_indices,
            view_ids=view_ids,
            angle_ids=angle_ids,
            depth_ids=depth_ids,
            points_m=points[point_indices],
            view_directions=view_directions,
            rotations=rotations,
            offsets=offsets_grid[view_ids, angle_ids, depth_ids],
        )
