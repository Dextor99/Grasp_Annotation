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


@dataclass(frozen=True)
class WidthGeometryEvaluation:
    """Result of the official max-width crop and adaptive-width pass."""

    width_m: float
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


def analyze_width(points_y_m: np.ndarray, hole_size_m: float = 0.018, loose_factor_m: float = 0.004) -> float:
    """Match the public GraspNet-style ``analyze_width`` implementation."""

    values = np.asarray(points_y_m, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("points_y_m must be a non-empty finite array")
    if float(hole_size_m) <= 0 or float(loose_factor_m) < 0:
        raise ValueError("hole_size_m must be positive and loose_factor_m cannot be negative")
    left = np.sort(-values[values <= 0])
    right = np.sort(values[values >= 0])
    if len(left) == 0 or len(right) == 0:
        compressed = left[-1] if len(right) == 0 else right[-1]
        return float(2.0 * compressed + loose_factor_m)
    half = float(max(left[-1], right[-1]))
    compressed = 2.0 * half + float(loose_factor_m)
    left_next = np.append(left[1:], half)
    right_next = np.append(right[1:], half)
    left_holes = (left_next - left) >= float(hole_size_m)
    right_holes = (right_next - right) >= float(hole_size_m)
    if not np.any(left_holes) or not np.any(right_holes):
        return float(compressed)
    left_start, left_end = left[left_holes], left_next[left_holes]
    right_start, right_end = right[right_holes], right_next[right_holes]
    overlap = np.minimum(left_end[:, None], right_end[None, :]) - np.maximum(left_start[:, None], right_start[None, :])
    matches = np.argwhere(overlap >= float(hole_size_m))
    if len(matches):
        i, j = matches[0]
        return float(2.0 * max(left_start[i], right_start[j]) + float(loose_factor_m))
    return float(compressed)


def evaluate_adaptive_width(
    points_m: np.ndarray,
    grasp_point_m: np.ndarray,
    rotation: np.ndarray,
    depth_m: float,
    *,
    max_width_m: float = 0.12,
    height_m: float = 0.02,
    depth_base_m: float = 0.02,
    finger_width_m: float = 0.01,
    bottom_thickness_m: float = 0.1,
    empty_thresh: int = 10,
    hole_size_m: float = 0.018,
    loose_factor_m: float = 0.004,
) -> WidthGeometryEvaluation:
    """Compute adaptive opening and final official-compatible masks.

    The first pass is always cropped at ``max_width_m``.  Width estimation is
    based only on points in that inner region, then the final finger/palm and
    empty checks are evaluated with the estimated opening.
    """

    if float(depth_m) <= 0 or float(max_width_m) <= 0:
        raise ValueError("depth_m and max_width_m must be positive")
    target = _local_points(points_m, grasp_point_m, rotation)
    z, x, y = target[:, 2], target[:, 0], target[:, 1]
    height = (z > -height_m / 2.0) & (z < height_m / 2.0)
    depth = (x > -depth_base_m) & (x < depth_m)
    max_left = y < -max_width_m / 2.0
    max_right = y > max_width_m / 2.0
    inner_max = height & depth & (~max_left) & (~max_right)
    inner_count = int(np.count_nonzero(inner_max))
    if inner_count < int(empty_thresh):
        return WidthGeometryEvaluation(0.0, False, True, inner_count)
    width = float(np.clip(analyze_width(target[inner_max, 1], hole_size_m, loose_factor_m), 0.0, max_width_m))
    mask3 = y > -(width / 2.0 + finger_width_m)
    mask5 = y < (width / 2.0 + finger_width_m)
    mask4 = y < -width / 2.0
    mask6 = y > width / 2.0
    mask7 = (x > -(depth_base_m + bottom_thickness_m)) & (x < -depth_base_m)
    left = height & depth & mask3 & mask4
    right = height & depth & mask5 & mask6
    bottom = height & mask3 & mask5 & mask7
    inner_final = height & depth & (~mask4) & (~mask6)
    final_inner_count = int(np.count_nonzero(inner_final))
    return WidthGeometryEvaluation(
        width_m=width,
        collision=bool(np.any(left | right | bottom)),
        empty=final_inner_count < int(empty_thresh),
        inner_count=final_inner_count,
    )


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
    left_finger = mask_height & mask_depth & (y > -(opening_m / 2.0 + finger_width_m)) & (y < -opening_m / 2.0)
    right_finger = mask_height & mask_depth & (y < opening_m / 2.0 + finger_width_m) & (y > opening_m / 2.0)
    palm = (
        mask_height
        & (y > -(opening_m / 2.0 + finger_width_m))
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
