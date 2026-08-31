"""Open3D visualization helpers for finalized grasp annotations.

This module intentionally consumes exported ``grasps.json`` records and does
not call the grasp-generation pipeline.  It therefore remains independent of
the frozen v1.0 algorithm.
"""

from __future__ import annotations

import colorsys
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from gripper_model import GripperModel, align_vector_to_z
from object_preprocess import prepare_object


def load_grasp_records(results_dir):
    """Load the finalized grasp list from an export directory."""
    path = Path(results_dir) / "grasps.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing grasp file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("grasps.json must contain a list")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("each grasp record must be a JSON object")
    return records


def load_meta(results_dir):
    """Load optional metadata from an export directory."""
    path = Path(results_dir) / "meta.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("meta.json must contain an object")
    return metadata


def record_to_transform(record):
    """Convert a finalized record's local-to-object pose to a 4x4 matrix."""
    try:
        translation = np.asarray(record["translation"], dtype=float)
        rotation = np.asarray(record["rotation_matrix"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid grasp pose: {error}") from error
    if translation.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("translation must be (3,) and rotation_matrix must be (3,3)")
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)):
        raise ValueError("grasp pose must contain finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("rotation_matrix must be a valid SO(3) matrix")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def select_grasps(records, topk=20, score_threshold=None):
    """Return score-ranked records after optional threshold and Top-K filters."""
    if topk is not None and (not isinstance(topk, (int, np.integer)) or topk < 0):
        raise ValueError("topk must be a non-negative integer or None")
    selected = list(records)
    if score_threshold is not None:
        if not np.isfinite(float(score_threshold)):
            raise ValueError("score_threshold must be finite")
        selected = [record for record in selected if float(record["score_total"]) >= score_threshold]
    selected.sort(key=lambda record: float(record["score_total"]), reverse=True)
    if topk is not None:
        selected = selected[: int(topk)]
    return selected


def group_color(group_id):
    """Return a deterministic HSV-derived RGB color for a group id."""
    hue = (int(group_id) * 0.618033988749895) % 1.0
    return list(colorsys.hsv_to_rgb(hue, 0.75, 0.95))


def score_color(score, score_min, score_max):
    """Map low scores to blue and high scores to red."""
    if score_max <= score_min:
        alpha = 1.0
    else:
        alpha = (float(score) - float(score_min)) / (float(score_max) - float(score_min))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return [alpha, 0.25, 1.0 - alpha]


def make_object_cloud(object_path):
    """Load the object with the exact scale/downsampling used by generation."""
    object_data = prepare_object(object_path)
    cloud = copy.deepcopy(object_data.cloud_down)
    cloud.paint_uniform_color([0.65, 0.65, 0.65])
    return cloud, object_data


def make_gripper_lineset(record, color=(0.9, 0.2, 0.2), finger_length=100.0, finger_thickness=5.0):
    """Build a lightweight line gripper using the project's local axes."""
    opening = float(record["opening_mm"])
    if not np.isfinite(opening) or opening < 0:
        raise ValueError("opening_mm must be finite and non-negative")
    half_open = (opening + float(finger_thickness)) / 2.0
    points_local = np.array(
        [[0.0, -half_open, 0.0], [0.0, half_open, 0.0],
         [0.0, -half_open, -float(finger_length)], [0.0, half_open, -float(finger_length)]],
        dtype=float,
    )
    lines = np.array([[0, 2], [1, 3], [2, 3]], dtype=np.int32)
    homogeneous = np.column_stack((points_local, np.ones(len(points_local))))
    points_world = (record_to_transform(record) @ homogeneous.T).T[:, :3]
    geometry = o3d.geometry.LineSet()
    geometry.points = o3d.utility.Vector3dVector(points_world)
    geometry.lines = o3d.utility.Vector2iVector(lines)
    geometry.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=float), (len(lines), 1))
    )
    return geometry


def make_gripper_meshes(record, color=None, include_axes=False):
    """Build the actual project gripper meshes at a finalized grasp pose."""
    gripper = GripperModel(opening=float(record["opening_mm"]))
    gripper.transform(record_to_transform(record))
    geometries = gripper.get_meshes()
    physical_meshes = geometries if include_axes else geometries[:3]
    if color is not None:
        for mesh in physical_meshes:
            mesh.paint_uniform_color(list(color))
    return physical_meshes


def make_sphere(center, radius=2.5, color=(1.0, 0.7, 0.0)):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(list(color))
    sphere.translate(np.asarray(center, dtype=float))
    return sphere


def make_arrow(origin, direction, length=25.0, color=(0.1, 0.8, 0.2)):
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if direction.shape != (3,) or not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("arrow direction must be a finite non-zero vector")
    length = float(length)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("arrow length must be positive and finite")
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=max(length * 0.025, 0.4),
        cone_radius=max(length * 0.05, 0.8),
        cylinder_height=length * 0.75,
        cone_height=length * 0.25,
    )
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color(list(color))
    transform = np.eye(4)
    transform[:3, :3] = align_vector_to_z(direction / norm)
    transform[:3, 3] = np.asarray(origin, dtype=float)
    arrow.transform(transform)
    return arrow


def make_approach_arrow(record, length=30.0):
    anchor = np.asarray(record["anchor_point"], dtype=float)
    direction = np.asarray(record["approach_direction"], dtype=float)
    direction /= np.linalg.norm(direction)
    return make_arrow(anchor - direction * float(length), direction, length, color=(0.1, 0.85, 0.2))


def make_normal_arrow(record, length=25.0):
    return make_arrow(
        record["anchor_point"], record["anchor_normal"], length, color=(0.15, 0.35, 1.0)
    )


def make_grasp_frame(record, size=15.0):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(size))
    frame.transform(record_to_transform(record))
    return frame


def _record_color(record, color_by, score_min, score_max):
    if color_by == "view":
        return group_color(record["view_id"])
    if color_by == "anchor":
        return group_color(int(record["view_id"]) * 100 + int(record["anchor_id"]))
    if color_by == "approach":
        return group_color(record["approach_id"])
    if color_by == "score":
        return score_color(record["score_total"], score_min, score_max)
    raise ValueError("color_by must be one of score, view, anchor, approach")


def build_visualization_geometries(
    object_path,
    records,
    mode="overlay",
    style="line",
    color_by="score",
    show_anchor=False,
    show_approach=False,
    show_normal=False,
    show_frame=False,
):
    """Compose object, grippers, and optional diagnostic markers."""
    if mode not in {"single", "overlay", "by_view", "by_anchor", "by_approach"}:
        raise ValueError("unsupported visualization mode")
    if style not in {"line", "mesh"}:
        raise ValueError("style must be line or mesh")
    cloud, object_data = make_object_cloud(object_path)
    geometries = [cloud]
    records = list(records)
    if not records:
        return geometries, object_data
    scores = [float(record["score_total"]) for record in records]
    score_min, score_max = min(scores), max(scores)
    for record in records:
        color = _record_color(record, color_by, score_min, score_max)
        if style == "mesh":
            geometries.extend(make_gripper_meshes(record, color=color))
        else:
            geometries.append(make_gripper_lineset(record, color=color))
        if show_anchor:
            geometries.append(make_sphere(record["anchor_point"]))
        if show_approach:
            geometries.append(make_approach_arrow(record))
        if show_normal:
            geometries.append(make_normal_arrow(record))
        if show_frame:
            geometries.append(make_grasp_frame(record))
    return geometries, object_data


def print_visualization_summary(all_records, shown_records, meta):
    """Print a compact summary useful for visual inspection logs."""
    print("\n=== Grasp visualization ===")
    print(f"Total unique grasps: {len(all_records)}")
    print(f"Displayed grasps: {len(shown_records)}")
    if shown_records:
        scores = [float(record["score_total"]) for record in shown_records]
        print(f"Displayed score range: [{min(scores):.6f}, {max(scores):.6f}]")
        print(f"View IDs: {sorted({int(record['view_id']) for record in shown_records})}")
        print(f"Anchor IDs: {sorted({int(record['anchor_id']) for record in shown_records})}")
        print(f"Approach IDs: {sorted({int(record['approach_id']) for record in shown_records})}")
    if meta:
        for label, key in (("Raw grasps", "raw_grasp_count"), ("Unique grasps", "unique_grasp_count")):
            if key in meta:
                print(f"{label}: {meta[key]}")
        if meta.get("merge_reduction_ratio") is not None:
            print(f"Merge reduction: {100.0 * float(meta['merge_reduction_ratio']):.2f}%")


def show_geometries(geometries, title="6-DoF Grasp Visualization", point_size=3.0, save_image=None):
    """Display geometries and optionally capture a screenshot after rendering."""
    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(window_name=title, width=1280, height=900):
        raise RuntimeError("Open3D could not create a visualization window")
    try:
        for geometry in geometries:
            visualizer.add_geometry(geometry)
        render_option = visualizer.get_render_option()
        render_option.point_size = float(point_size)
        render_option.background_color = np.array([1.0, 1.0, 1.0])
        visualizer.poll_events()
        visualizer.update_renderer()
        if save_image is not None:
            image_path = Path(save_image)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not visualizer.capture_screen_image(str(image_path), do_render=True):
                raise RuntimeError(f"Open3D failed to save screenshot: {image_path}")
        visualizer.run()
    finally:
        visualizer.destroy_window()
