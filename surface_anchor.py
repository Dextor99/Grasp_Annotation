"""Sparse representative anchors and smoothed local surface normals."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SurfaceAnchor:
    anchor_id: int
    point: np.ndarray
    normal: np.ndarray


def _validate_surface(points, normals=None):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("surface points must have non-empty shape (N, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("surface points must be finite")
    if normals is None:
        return points, None
    normals = np.asarray(normals, dtype=float)
    if normals.shape != points.shape or not np.all(np.isfinite(normals)):
        raise ValueError("surface normals must be finite and match points shape")
    return points, normals


def farthest_point_sample_indices(points, num_samples):
    """Select deterministic farthest-point-sampling indices."""
    points, _ = _validate_surface(points)
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    count = min(num_samples, len(points))
    centroid = np.mean(points, axis=0)
    selected = [int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))]
    selected_mask = np.zeros(len(points), dtype=bool)
    selected_mask[selected[0]] = True
    min_distances = np.sum((points - points[selected[0]]) ** 2, axis=1)
    while len(selected) < count:
        scores = min_distances.copy()
        scores[selected_mask] = -np.inf
        next_index = int(np.argmax(scores))
        if not np.isfinite(scores[next_index]) or scores[next_index] <= 1e-12:
            break
        selected.append(next_index)
        selected_mask[next_index] = True
        distances = np.sum((points - points[next_index]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distances)
    return np.asarray(selected, dtype=int)


def estimate_local_normal(points, normals, anchor_point, view_direction, normal_knn=30):
    """Average KNN normals, normalize, and orient them toward the virtual camera."""
    points, normals = _validate_surface(points, normals)
    anchor = np.asarray(anchor_point, dtype=float)
    view = np.asarray(view_direction, dtype=float)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise ValueError("anchor_point must be a finite 3-vector")
    view_norm = np.linalg.norm(view)
    if view.shape != (3,) or not np.isfinite(view_norm) or view_norm <= 1e-12:
        raise ValueError("view_direction must be a non-zero finite 3-vector")
    if not isinstance(normal_knn, int) or normal_knn < 1:
        raise ValueError("normal_knn must be a positive integer")
    neighbor_count = min(normal_knn, len(points))
    distances = np.sum((points - anchor) ** 2, axis=1)
    neighbors = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
    local_normal = np.sum(normals[neighbors], axis=0)
    length = np.linalg.norm(local_normal)
    if not np.isfinite(length) or length <= 1e-12:
        local_normal = normals[int(np.argmin(distances))].copy()
        length = np.linalg.norm(local_normal)
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError("local normals cancel to a zero vector")
    local_normal /= length
    view /= view_norm
    if float(local_normal @ view) < 0.0:
        local_normal = -local_normal
    return local_normal


def build_surface_anchors(
    surface_points,
    surface_normals,
    view_direction,
    num_anchors=3,
    normal_knn=30,
):
    """Build FPS anchors with KNN-smoothed normals for one visible surface."""
    points, normals = _validate_surface(surface_points, surface_normals)
    indices = farthest_point_sample_indices(points, num_anchors)
    return [
        SurfaceAnchor(
            anchor_id=anchor_id,
            point=points[index].copy(),
            normal=estimate_local_normal(
                points,
                normals,
                points[index],
                view_direction,
                normal_knn=normal_knn,
            ),
        )
        for anchor_id, index in enumerate(indices)
    ]
