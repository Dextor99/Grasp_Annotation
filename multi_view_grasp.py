"""Generate, score, and aggregate grasps across virtual camera views.

Surface filtering is normal-based front-facing filtering, so it does not model
self-occlusion or provide true ray-cast visibility.
"""

from dataclasses import dataclass
import math

import numpy as np
import open3d as o3d

from grasp_detect import grasp_detect_from_surface
from grasp_merge import merge_duplicate_grasps
from grasp_score_V3 import compute_grasp_scores_simple
from surface_visibility import filter_front_facing_surface
from view_sampling import generate_viewpoints


@dataclass
class MultiViewResult:
    grasps: list[dict]
    skipped_views: list[dict]
    view_candidate_counts: dict[int, int]


def _load_cloud(path):
    """Read a PLY once, retaining supplied normals or estimating missing ones."""
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise ValueError("PLY must contain at least one point with shape (N, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("PLY points must be finite")

    normals = np.asarray(cloud.normals, dtype=float)
    if normals.shape != points.shape:
        cloud.estimate_normals()
        normals = np.asarray(cloud.normals, dtype=float)
    if normals.shape != points.shape or not np.all(np.isfinite(normals)):
        raise ValueError("PLY normals must have shape (N, 3) and be finite")
    return points, normals


def _full_cloud(points, normals):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud


def _finite_score(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _score_total(grasp):
    force_closure = _finite_score(grasp.get("score_force_closure"))
    if force_closure is not None:
        return force_closure
    inner_points_ratio = _finite_score(grasp.get("score_inner_points_ratio"))
    if inner_points_ratio is not None:
        return inner_points_ratio
    return float("-inf")


def generate_multi_view_grasps(
    path,
    num_views=60,
    position_threshold_mm=5.0,
    rotation_threshold_deg=10.0,
    *,
    loader=_load_cloud,
    detector=grasp_detect_from_surface,
    scorer=compute_grasp_scores_simple,
    deduplicate=True,
):
    """Produce view-conditioned grasp candidates and optionally deduplicate them."""
    points, normals = loader(path)
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or normals.shape != points.shape:
        raise ValueError("loader must return points and normals with shape (N, 3)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normals)):
        raise ValueError("loader must return finite points and normals")

    cloud = _full_cloud(points, normals)
    grasps = []
    skipped_views = []
    view_candidate_counts = {}

    for view_id, view in enumerate(generate_viewpoints(num_views)):
        visible_points, visible_normals, _ = filter_front_facing_surface(points, normals, view)
        if len(visible_points) == 0:
            view_candidate_counts[view_id] = 0
            skipped_views.append({"view_id": view_id, "reason": "no_front_facing_points"})
            continue

        normalized_view = view / np.linalg.norm(view)
        candidates = detector(
            visible_points, visible_normals, normalized_view, {"view_id": view_id}
        )
        candidates = [] if candidates is None else list(candidates)
        view_candidate_counts[view_id] = len(candidates)
        if not candidates:
            skipped_views.append({"view_id": view_id, "reason": "no_candidates"})
            continue

        scored = scorer(candidates, cloud)
        scored = candidates if scored is None else scored
        for candidate in scored:
            if not isinstance(candidate, dict):
                continue
            grasp = dict(candidate)
            grasp["view_id"] = view_id
            grasp["view_direction"] = normalized_view.copy()
            grasp["score_total"] = _score_total(grasp)
            grasps.append(grasp)

    if deduplicate:
        grasps = merge_duplicate_grasps(
            grasps, position_threshold_mm, rotation_threshold_deg
        )
    return MultiViewResult(grasps, skipped_views, view_candidate_counts)
