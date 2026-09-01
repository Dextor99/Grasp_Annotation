"""Formal V4 grasp quality scoring for offline re-ranking.

The frozen generation and legacy V3 score are intentionally left untouched.
This module extracts bilateral contact features from an exported grasp pose and
adds a separate V4 score so the two rankings can be compared safely.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


DEFAULT_FINGER_THICKNESS_MM = 5.0
DEFAULT_FINGER_LENGTH_MM = 100.0
DEFAULT_SUPPORT_REFERENCE_AREA_MM2 = 15.0 * DEFAULT_FINGER_LENGTH_MM
DEFAULT_STABILITY_SCALE = 0.6


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(length) or length < 1e-12:
        return np.zeros(3, dtype=float)
    return vector / length


def _support_area(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points[:, [0, 2]]).volume)
    except Exception:
        return 0.0


def estimate_voxel_size_mm(object_data, max_points=2000):
    """Estimate point spacing used for contact-band width.

    ``ObjectData`` does not require a voxel-size field, so use a deterministic
    nearest-neighbour estimate and keep a conservative finite fallback.
    """
    points = np.asarray(getattr(object_data, "points", []), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        return 1.0
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        sample = points[indices]
    else:
        sample = points
    distances = cKDTree(points).query(sample, k=2)[0][:, 1]
    finite = distances[np.isfinite(distances) & (distances > 1e-9)]
    if len(finite) == 0:
        return 1.0
    return float(np.median(finite))


def compose_v4_score(normal, support, stability):
    """Compose bounded V4 components with the frozen unweighted geometric mean."""
    components = np.asarray([normal, support, stability], dtype=float)
    if components.shape != (3,) or not np.all(np.isfinite(components)):
        raise ValueError("V4 components must be three finite values")
    components = np.clip(components, 0.0, 1.0)
    return float(np.clip(np.prod(components) ** (1.0 / 3.0), 0.0, 1.0))


def stability_from_center_distance(distance_normalized, stability_scale=DEFAULT_STABILITY_SCALE):
    """Return the soft centrality prior used by the formal scorer."""
    distance_normalized = float(distance_normalized)
    stability_scale = float(stability_scale)
    if not np.isfinite(distance_normalized) or not np.isfinite(stability_scale) or stability_scale <= 0:
        raise ValueError("distance and stability scale must be finite; scale must be positive")
    return float(np.exp(-((distance_normalized / stability_scale) ** 2)))


def _empty_metrics(origin, object_data):
    center = np.asarray(getattr(object_data, "center", np.zeros(3)), dtype=float)
    distance = float(np.linalg.norm(origin - center))
    radius = float(getattr(object_data, "radius", 0.0))
    normalized = distance / radius if radius > 1e-9 else 0.0
    stability = float(np.exp(-((normalized / DEFAULT_STABILITY_SCALE) ** 2)))
    return {
        "contact_points_left": 0,
        "contact_points_right": 0,
        "contact_area_left_mm2": 0.0,
        "contact_area_right_mm2": 0.0,
        "normal_alignment_left": 0.0,
        "normal_alignment_right": 0.0,
        "normal_alignment_robust": 0.0,
        "normal_dispersion_left": 0.0,
        "normal_dispersion_right": 0.0,
        "score_v4_normal": 0.0,
        "score_v4_support": 0.0,
        "score_v4_stability": stability,
        "score_v4_normal_dispersion": 0.0,
        "center_distance_mm": distance,
        "center_distance_normalized": normalized,
        "contact_band_mm": 0.0,
        "score_total_v4": 0.0,
    }


def _robust_side_score(normals, target):
    """Return median non-negative alignment and a high-is-consistent score."""
    normals = np.asarray(normals, dtype=float)
    if normals.ndim != 2 or normals.shape[1] != 3 or len(normals) == 0:
        return 0.0, 0.0, 0.0
    normalized = np.asarray([_unit(normal) for normal in normals])
    target = _unit(target)
    alignments = np.clip(normalized @ target, -1.0, 1.0)
    robust = float(np.clip(np.median(alignments), 0.0, 1.0))
    # Median direction makes this dispersion resistant to a few bad normals.
    median_normal = _unit(np.median(normalized, axis=0))
    if np.linalg.norm(median_normal) < 1e-12:
        dispersion_score = 0.0
    else:
        consistency = np.clip(normalized @ median_normal, -1.0, 1.0)
        dispersion_score = float(np.clip(np.median(consistency), 0.0, 1.0))
    return robust, 1.0 - dispersion_score, dispersion_score


def score_grasp_v4(
    record,
    object_data,
    *,
    voxel_size_mm=None,
    finger_thickness_mm=DEFAULT_FINGER_THICKNESS_MM,
    finger_length_mm=DEFAULT_FINGER_LENGTH_MM,
    support_reference_area_mm2=DEFAULT_SUPPORT_REFERENCE_AREA_MM2,
    stability_scale=DEFAULT_STABILITY_SCALE,
):
    """Return a copy of ``record`` augmented with formal V4 metrics.

    Contact bands use ``delta = 2.5 * voxel_size_mm``.  The total is the
    unweighted geometric mean of robust bilateral normal alignment, bilateral
    support and soft object-centre stability.  Existing ``score_total`` is
    copied to ``score_total_v3`` and is never overwritten.
    """
    points_world = np.asarray(getattr(object_data, "points", []), dtype=float)
    normals_world = np.asarray(getattr(object_data, "normals", []), dtype=float)
    if points_world.ndim != 2 or points_world.shape[1] != 3 or points_world.shape != normals_world.shape:
        raise ValueError("object_data points and normals must both have shape (N, 3)")
    try:
        translation = np.asarray(record["translation"], dtype=float)
        rotation_pose = np.asarray(record["rotation_matrix"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid grasp pose: {error}") from error
    if translation.shape != (3,) or rotation_pose.shape != (3, 3):
        raise ValueError("translation must have shape (3,) and rotation_matrix (3,3)")
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation_pose)):
        raise ValueError("grasp pose must be finite")
    object_world = np.asarray(getattr(object_data, "T_object_world", np.eye(4)), dtype=float)
    if object_world.shape != (4, 4) or not np.all(np.isfinite(object_world)):
        raise ValueError("object_data.T_object_world must be a finite (4,4) matrix")
    pose_world = object_world @ np.block(
        [[rotation_pose, translation[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]
    )
    rotation = pose_world[:3, :3]
    origin = pose_world[:3, 3]
    points_local = (rotation.T @ (points_world - origin).T).T
    normals_local = (rotation.T @ normals_world.T).T
    opening = float(record.get("opening_mm", record.get("grasp_width_mm", 0.0)))
    opening = max(opening, 0.0)
    inner_mask = (
        (np.abs(points_local[:, 0]) <= float(finger_thickness_mm) / 2.0)
        & (np.abs(points_local[:, 1]) <= opening / 2.0)
        & (points_local[:, 2] >= -float(finger_length_mm))
        & (points_local[:, 2] <= 0.0)
    )

    result = dict(record)
    old_score = record.get("score_total", record.get("score_total_v3", 0.0))
    result["score_total_v3"] = float(old_score) if isinstance(old_score, Real) and np.isfinite(float(old_score)) else 0.0
    if np.count_nonzero(inner_mask) < 2:
        result.update(_empty_metrics(origin, object_data))
        return result

    inner_points = points_local[inner_mask]
    inner_normals = normals_local[inner_mask]
    y_values = inner_points[:, 1]
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    span = max(y_max - y_min, 1e-9)
    if voxel_size_mm is None:
        voxel_size_mm = estimate_voxel_size_mm(object_data)
    voxel_size_mm = float(voxel_size_mm)
    if not np.isfinite(voxel_size_mm) or voxel_size_mm <= 0:
        voxel_size_mm = 1.0
    contact_band = max(2.5, 2.5 * voxel_size_mm)
    # Keep the band meaningful for very thin local spans without changing the
    # two-sided definition.
    contact_band = min(contact_band, span / 2.0 if span > 0 else contact_band)
    left_mask = y_values <= y_min + contact_band
    right_mask = y_values >= y_max - contact_band
    left_points, right_points = inner_points[left_mask], inner_points[right_mask]
    left_normals, right_normals = inner_normals[left_mask], inner_normals[right_mask]
    left_alignment, left_dispersion, left_dispersion_score = _robust_side_score(
        left_normals, [0.0, -1.0, 0.0]
    )
    right_alignment, right_dispersion, right_dispersion_score = _robust_side_score(
        right_normals, [0.0, 1.0, 0.0]
    )
    normal_score = float(np.clip(min(left_alignment, right_alignment), 0.0, 1.0))
    normal_dispersion_score = float(np.clip(min(left_dispersion_score, right_dispersion_score), 0.0, 1.0))
    reference_area = max(float(support_reference_area_mm2), 1e-9)
    left_area = _support_area(left_points)
    right_area = _support_area(right_points)
    left_support = float(np.clip(left_area / reference_area, 0.0, 1.0))
    right_support = float(np.clip(right_area / reference_area, 0.0, 1.0))
    support_score = float(math.sqrt(left_support * right_support))
    center = np.asarray(getattr(object_data, "center", np.zeros(3)), dtype=float)
    center_distance = float(np.linalg.norm(origin - center))
    radius = float(getattr(object_data, "radius", 0.0))
    center_distance_normalized = center_distance / radius if radius > 1e-9 else 0.0
    stability = float(np.exp(-((center_distance_normalized / max(float(stability_scale), 1e-9)) ** 2)))
    total = compose_v4_score(normal_score, support_score, stability)
    result.update(
        {
            "contact_points_left": int(np.count_nonzero(left_mask)),
            "contact_points_right": int(np.count_nonzero(right_mask)),
            "contact_area_left_mm2": left_area,
            "contact_area_right_mm2": right_area,
            "normal_alignment_left": left_alignment,
            "normal_alignment_right": right_alignment,
            "normal_alignment_robust": normal_score,
            "normal_dispersion_left": left_dispersion,
            "normal_dispersion_right": right_dispersion,
            "score_v4_normal": normal_score,
            "score_v4_support": support_score,
            "score_v4_stability": stability,
            "score_v4_normal_dispersion": normal_dispersion_score,
            "center_distance_mm": center_distance,
            "center_distance_normalized": center_distance_normalized,
            "contact_band_mm": contact_band,
            "score_total_v4": total,
        }
    )
    return result


def score_grasps_v4(records, object_data, **kwargs):
    """Score and rank records by V4 while preserving the V3 score field."""
    scored = [score_grasp_v4(record, object_data, **kwargs) for record in records]
    return sorted(scored, key=lambda record: float(record["score_total_v4"]), reverse=True)
