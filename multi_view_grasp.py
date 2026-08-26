"""Minimal multi-view surface-conditioned grasp generation pipeline."""

from __future__ import annotations

import numpy as np

from grasp_detect import grasp_detect_from_surface
from object_preprocess import prepare_object
from surface_visibility import select_front_facing_surface
from view_sampling import fibonacci_directions


def generate_multi_view_grasps(
    ply_path,
    num_views=5,
    visibility_threshold=0.0,
    enable_visualization=False,
    object_data=None,
):
    """Prepare one object, then generate view-conditioned grasps for each view."""
    object_data = object_data or prepare_object(ply_path)
    views = fibonacci_directions(num_views)
    all_grasps = []
    for view_id, view in enumerate(views):
        surface_points, surface_normals = select_front_facing_surface(
            object_data.points,
            object_data.normals,
            view,
            threshold=visibility_threshold,
        )
        if len(surface_points) == 0:
            print(f"View {view_id}: surface 0 -> grasps 0")
            continue
        grasps = grasp_detect_from_surface(
            object_data,
            surface_points,
            surface_normals,
            view,
            enable_visualization=enable_visualization,
        )
        for grasp in grasps:
            grasp["view_id"] = view_id
            grasp["view_direction"] = view.tolist()
        all_grasps.extend(grasps)
        print(f"View {view_id}: surface {len(surface_points)} -> grasps {len(grasps)}")
    print(f"Total raw grasps: {len(all_grasps)}")
    return all_grasps
