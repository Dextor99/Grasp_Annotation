"""Raw GraspNet-compatible label arrays and compact valid-grasp records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidate_generation import CandidateBatch
from .config import DenseAnnotationConfig
from .view_sampling import generate_view_rotations, generate_views


@dataclass
class RawLabelArrays:
    points: np.ndarray
    offsets: np.ndarray
    collision: np.ndarray
    scores: np.ndarray

    @classmethod
    def create(cls, point_count: int, config: DenseAnnotationConfig) -> "RawLabelArrays":
        if int(point_count) < 0:
            raise ValueError("point_count must be non-negative")
        shape = (int(point_count), config.num_views, config.num_angles, len(config.depths_m))
        return cls(
            points=np.zeros((int(point_count), 3), dtype=np.float32),
            offsets=np.zeros((*shape, 3), dtype=np.float32),
            collision=np.ones(shape, dtype=bool),
            scores=np.full(shape, -1.0, dtype=np.float32),
        )

    def write_point_result(self, point_index: int, batch: CandidateBatch, widths_m: np.ndarray, collision: np.ndarray, mu_min: np.ndarray) -> None:
        if batch.size == 0 or not np.all(batch.point_indices == int(point_index)):
            raise ValueError("batch must contain only the requested point")
        shape = self.collision.shape[1:]
        widths = np.asarray(widths_m, dtype=np.float32).reshape(shape)
        collisions = np.asarray(collision, dtype=bool).reshape(shape)
        scores = np.asarray(mu_min, dtype=np.float32).reshape(shape)
        if not np.isfinite(widths).all() or not np.isfinite(scores[~collisions & (scores >= 0)]).all():
            raise ValueError("widths and valid scores must be finite")
        self.offsets[int(point_index), ..., 0:2] = batch.offsets.reshape((*shape, 3))[..., 0:2]
        self.offsets[int(point_index), ..., 2] = widths
        self.collision[int(point_index)] = collisions
        self.scores[int(point_index)] = np.where(collisions, -1.0, scores)

    def to_valid_grasps(self, config: DenseAnnotationConfig) -> np.ndarray:
        """Return `(K, 17)` records: quality, width, height, depth, R, t, object id."""

        if self.points.shape[0] == 0:
            return np.empty((0, 17), dtype=np.float32)
        views = generate_views(config.num_views)
        rotations = generate_view_rotations(
            np.repeat(views, config.num_angles, axis=0),
            np.tile(np.arange(config.num_angles, dtype=np.float32) * (np.pi / config.num_angles), config.num_views),
        ).reshape(config.num_views, config.num_angles, 3, 3)
        valid = np.argwhere((~self.collision) & np.isfinite(self.scores) & (self.scores >= 0.0))
        records = np.empty((len(valid), 17), dtype=np.float32)
        depths = np.asarray(config.depths_m, dtype=np.float32)
        for row, (point_id, view_id, angle_id, depth_id) in enumerate(valid):
            rotation = rotations[view_id, angle_id]
            depth = float(depths[depth_id])
            translation = self.points[point_id] + rotation @ np.array([depth, 0.0, 0.0], dtype=np.float32)
            records[row, 0] = 1.1 - self.scores[point_id, view_id, angle_id, depth_id]
            records[row, 1] = self.offsets[point_id, view_id, angle_id, depth_id, 2]
            records[row, 2] = config.height_m
            records[row, 3] = depth
            records[row, 4:13] = rotation.reshape(9)
            records[row, 13:16] = translation
            records[row, 16] = 0.0
        return records
