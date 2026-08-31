"""Minimal multi-view surface-conditioned grasp generation pipeline."""

from __future__ import annotations

import numpy as np

from approach_sampling import sample_normal_guided_approaches
from grasp_detect import grasp_detect_from_anchor_approach, grasp_detect_from_surface
from object_preprocess import prepare_object
from surface_anchor import build_surface_anchors
from surface_visibility import select_front_facing_surface
from view_sampling import fibonacci_directions


def generate_multi_view_grasps(
    ply_path,
    num_views=5,
    visibility_threshold=0.0,
    enable_visualization=False,
    object_data=None,
    mode="global",
    num_anchors_per_view=3,
    normal_knn=30,
    cone_angle_deg=15.0,
    num_approach_azimuth=4,
):
    """Generate global, local-normal, or local-cone multi-view grasps."""
    if mode not in {"global", "normal", "cone"}:
        raise ValueError("mode must be 'global', 'normal', or 'cone'")
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

        view_grasps = []
        if mode == "global":
            surface_center = np.mean(surface_points, axis=0)
            surface_normal = np.sum(surface_normals, axis=0)
            normal_length = np.linalg.norm(surface_normal)
            if normal_length > 1e-12:
                surface_normal /= normal_length
            else:
                surface_normal = view.copy()
            grasps = grasp_detect_from_surface(
                object_data,
                surface_points,
                surface_normals,
                view,
                enable_visualization=enable_visualization,
            )
            metadata = {
                "view_id": view_id,
                "view_direction": view.tolist(),
                "anchor_id": -1,
                "anchor_point": surface_center.tolist(),
                "anchor_normal": surface_normal.tolist(),
                "approach_id": 0,
                "approach_direction": (-view).tolist(),
                "approach_offset_deg": 0.0,
            }
            for grasp in grasps:
                grasp.update(metadata)
            view_grasps.extend(grasps)
        else:
            anchors = build_surface_anchors(
                surface_points,
                surface_normals,
                view,
                num_anchors=num_anchors_per_view,
                normal_knn=normal_knn,
            )
            for anchor in anchors:
                approaches = sample_normal_guided_approaches(
                    anchor.normal,
                    cone_angle_deg=cone_angle_deg,
                    num_azimuth=num_approach_azimuth,
                )
                if mode == "normal":
                    approaches = approaches[:1]
                for approach in approaches:
                    metadata = {
                        "view_id": view_id,
                        "view_direction": view.tolist(),
                        "anchor_id": anchor.anchor_id,
                        "anchor_point": anchor.point.tolist(),
                        "anchor_normal": anchor.normal.tolist(),
                        "approach_id": approach.approach_id,
                        "approach_direction": approach.direction.tolist(),
                        "approach_offset_deg": approach.offset_deg,
                    }
                    grasps = grasp_detect_from_anchor_approach(
                        object_data=object_data,
                        anchor_point=anchor.point,
                        anchor_normal=anchor.normal,
                        approach_direction=approach.direction,
                        metadata=metadata,
                        enable_visualization=enable_visualization,
                    )
                    for grasp in grasps:
                        grasp.update(metadata)
                    view_grasps.extend(grasps)

        all_grasps.extend(view_grasps)
        print(f"View {view_id}: surface {len(surface_points)} -> grasps {len(view_grasps)}")
    print(f"Total raw grasps: {len(all_grasps)}")
    return all_grasps
