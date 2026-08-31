"""Final local closure refinement for collision-feasible grasp candidates."""

from __future__ import annotations

import numpy as np


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
    if valid_points and "T_gripper_object" in refined:
        pose = np.asarray(refined["T_gripper_object"], dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("T_gripper_object must be a finite (4,4) matrix")
        y_min = float(np.min(points[:, 1]))
        y_max = float(np.max(points[:, 1]))
        center_offset = (y_min + y_max) / 2.0
        grasp_width = (y_max - y_min) + 2.0 * margin_mm
        pose = pose.copy()
        pose[:3, 3] += pose[:3, :3][:, 1] * center_offset
        refined["T_gripper_object"] = pose
        refined["closure_center_offset_mm"] = center_offset
        refined["grasp_width_mm"] = grasp_width
        refined["closure_refined"] = True
    else:
        refined["closure_center_offset_mm"] = 0.0
        refined["grasp_width_mm"] = 0.0 if opening is None else opening
        refined["closure_refined"] = False
    refined["closure_margin_mm"] = margin_mm
    return refined


def refine_grasp_closures(grasps, margin_mm=2.0):
    """Apply :func:`refine_grasp_closure` without mutating input candidates."""
    return [refine_grasp_closure(grasp, margin_mm=margin_mm) for grasp in grasps]
