"""Official-compatible parallel-jaw gripper geometry masks.

The local-frame inequalities mirror the collision and empty-region masks in
GraspNetAPI's evaluation utilities.  They require only object points, so the
Gate 4 sanity tests do not depend on an SDF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometryEvaluation:
    collision: bool
    empty: bool
    inner_count: int


def _local_points(points_m: np.ndarray, grasp_point_m: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float32)
    point = np.asarray(grasp_point_m, dtype=np.float32)
    matrix = np.asarray(rotation, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_m must have shape (N, 3), got {points.shape}")
    if point.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("grasp_point_m must be (3,) and rotation must be (3, 3)")
    if not np.isfinite(points).all() or not np.isfinite(point).all() or not np.isfinite(matrix).all():
        raise ValueError("geometry inputs must be finite")
    return (points - point[None, :]) @ matrix


def estimate_opening_m(
    points_m: np.ndarray,
    grasp_point_m: np.ndarray,
    rotation: np.ndarray,
    max_width_m: float,
    margin_m: float = 0.0,
) -> float:
    """Estimate a bounded opening from the local closing-axis extent."""

    if max_width_m <= 0 or margin_m < 0:
        raise ValueError("max_width_m must be positive and margin_m cannot be negative")
    local = _local_points(points_m, grasp_point_m, rotation)
    if len(local) == 0:
        return 0.0
    span = float(local[:, 1].max() - local[:, 1].min()) + 2.0 * float(margin_m)
    return float(np.clip(span, 0.0, max_width_m))


def evaluate_gripper_geometry(
    points_m: np.ndarray,
    grasp_point_m: np.ndarray,
    rotation: np.ndarray,
    opening_m: float,
    depth_m: float,
    height_m: float,
    depth_base_m: float,
    finger_width_m: float,
    empty_thresh: int,
) -> GeometryEvaluation:
    """Evaluate collision and empty masks in the official gripper frame."""

    values = {
        "opening_m": opening_m,
        "depth_m": depth_m,
        "height_m": height_m,
        "depth_base_m": depth_base_m,
        "finger_width_m": finger_width_m,
    }
    if any(float(value) <= 0 for value in values.values()):
        raise ValueError("gripper dimensions must be positive")
    if int(empty_thresh) != empty_thresh or empty_thresh < 0:
        raise ValueError("empty_thresh must be a non-negative integer")

    target = _local_points(points_m, grasp_point_m, rotation)
    z = target[:, 2]
    x = target[:, 0]
    y = target[:, 1]
    mask_height = (z > -height_m / 2.0) & (z < height_m / 2.0)
    mask_depth = (x > -depth_base_m) & (x < depth_m)
    left_finger = mask_height & mask_depth & (y < -(opening_m / 2.0 + finger_width_m)) & (y < -opening_m / 2.0)
    right_finger = mask_height & mask_depth & (y < opening_m / 2.0 + finger_width_m) & (y > opening_m / 2.0)
    palm = (
        mask_height
        & (y < -(opening_m / 2.0 + finger_width_m))
        & (y < opening_m / 2.0 + finger_width_m)
        & (x > -(depth_base_m + finger_width_m))
        & (x < -depth_base_m)
    )
    inner = mask_height & mask_depth & (y >= -opening_m / 2.0) & (y <= opening_m / 2.0)
    inner_count = int(np.count_nonzero(inner))
    return GeometryEvaluation(
        collision=bool(np.any(left_finger | right_finger | palm)),
        empty=inner_count < int(empty_thresh),
        inner_count=inner_count,
    )
