"""Final local closure refinement for collision-feasible grasp candidates."""

from __future__ import annotations

import numpy as np

from grasp_detect import CollisionIndex, check_collision
from gripper_model import create_gripper_model


def refine_grasp_closure(grasp, margin_mm=2.0):
    """Center a grasp on its inner-point span and derive final jaw width.

    ``opening`` remains the aperture used during collision search.  The
    refined physical width is exposed separately as ``grasp_width_mm`` so
    search and execution parameters are not conflated.
    """
    margin_mm = float(margin_mm)
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("margin_mm must be finite and non-negative")
    # Candidate records contain Open3D meshes; keep those shared and copy only
    # the pose we mutate below to avoid duplicating large geometry objects.
    refined = dict(grasp)
    opening = refined.get("opening", refined.get("search_opening_mm"))
    if opening is not None:
        opening = float(opening)
        if not np.isfinite(opening) or opening < 0:
            raise ValueError("opening must be finite and non-negative")
        refined["search_opening_mm"] = opening

    points = np.asarray(refined.get("inner_points_local", np.empty((0, 3))), dtype=float)
    valid_points = points.ndim == 2 and points.shape[1] == 3 and len(points) >= 2 and np.all(
        np.isfinite(points)
    )
    refined["requested_margin_mm"] = margin_mm
    refined["score_y0_diff_before_refinement"] = refined.get(
        "score_y0_diff_before_refinement", refined.get("score_y0_diff")
    )
    if valid_points and "T_gripper_object" in refined:
        pose = np.asarray(refined["T_gripper_object"], dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("T_gripper_object must be a finite (4,4) matrix")
        y_min = float(np.min(points[:, 1]))
        y_max = float(np.max(points[:, 1]))
        support_span = y_max - y_min
        center_offset = (y_min + y_max) / 2.0
        refined["support_span_mm"] = support_span
        if opening is not None and support_span > opening + 1e-9:
            # The candidate is internally inconsistent: its support points do
            # not fit inside the aperture that was collision-checked.
            refined["closure_center_offset_mm"] = 0.0
            refined["grasp_width_mm"] = opening
            refined["effective_margin_mm"] = 0.0
            refined["closure_geometry_valid"] = False
            refined["closure_refined"] = False
        else:
            grasp_width = support_span + 2.0 * margin_mm
            if opening is not None:
                grasp_width = min(grasp_width, opening)
            effective_margin = max(0.0, (grasp_width - support_span) / 2.0)
            pose = pose.copy()
            pose[:3, 3] += pose[:3, :3][:, 1] * center_offset
            refined["T_gripper_object"] = pose
            centered_points = points.copy()
            centered_points[:, 1] -= center_offset
            refined["inner_points_local"] = centered_points
            refined["closure_center_offset_mm"] = center_offset
            refined["grasp_width_mm"] = grasp_width
            refined["effective_margin_mm"] = effective_margin
            refined["closure_geometry_valid"] = True
            refined["closure_refined"] = True
            refined["score_y0_diff_refined"] = abs(
                (y_min - center_offset) + (y_max - center_offset)
            )
    else:
        refined["closure_center_offset_mm"] = 0.0
        refined["grasp_width_mm"] = 0.0 if opening is None else opening
        refined["support_span_mm"] = 0.0
        refined["effective_margin_mm"] = 0.0
        refined["closure_geometry_valid"] = True
        refined["score_y0_diff_refined"] = 0.0
        refined["closure_refined"] = False
    refined["closure_margin_mm"] = margin_mm
    return refined


def refine_grasp_closures(grasps, margin_mm=2.0):
    """Apply :func:`refine_grasp_closure` without mutating input candidates."""
    return [refine_grasp_closure(grasp, margin_mm=margin_mm) for grasp in grasps]


def _rebuild_search_meshes(grasp, T_object_world):
    """Build the original search-aperture gripper at the refined pose."""
    pose_object = np.asarray(grasp["T_gripper_object"], dtype=float)
    object_to_world = np.asarray(T_object_world, dtype=float)
    if pose_object.shape != (4, 4) or object_to_world.shape != (4, 4):
        raise ValueError("grasp and object transforms must be 4x4 matrices")
    opening = float(grasp.get("search_opening_mm", grasp.get("opening")))
    return create_gripper_model(
        pose=object_to_world @ pose_object,
        opening=opening,
        finger_length=100.0,
    )["meshes"]


def validate_refined_grasp_closures(
    grasps,
    point_cloud,
    T_object_world,
    threshold_mm=3.0,
):
    """Recheck the refined center with the original search aperture.

    The final physical width is intentionally not used for this test: closing
    fingers onto contact points is expected.  Only the collision-safe search
    aperture is rebuilt and checked against the full object cloud.
    """
    collision_index = CollisionIndex.from_point_cloud(point_cloud)
    validated = []
    for grasp in grasps:
        candidate = dict(grasp)
        if not candidate.get("closure_geometry_valid", True):
            candidate["closure_pose_valid"] = False
            continue
        meshes = _rebuild_search_meshes(candidate, T_object_world)
        collision = check_collision(
            meshes,
            point_cloud,
            threshold=threshold_mm,
            collision_index=collision_index,
        )
        candidate["closure_pose_valid"] = not collision
        if not collision:
            candidate["meshes"] = meshes
            validated.append(candidate)
    return validated
