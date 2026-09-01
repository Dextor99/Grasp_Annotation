"""Interactive Open3D sanity view for common-evaluator Top-K grasps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.graspnet_annotation.config import DenseAnnotationConfig
from baselines.graspnet_annotation.view_sampling import generate_view_rotations, generate_views
from grasp_visualization import make_gripper_lineset, make_object_cloud, show_geometries
from scripts.common_eval.ours_official_common_eval import OURS_TO_OFFICIAL_AXES


def object_loader_argument(path: str | Path) -> str:
    """Return a plain string for Open3D APIs that reject ``Path`` objects."""

    return str(path)


def official_candidate_to_record(
    point_m: np.ndarray,
    rotation_official: np.ndarray,
    depth_m: float,
    width_m: float,
    object_world: np.ndarray,
    *,
    score_total: float = 0.0,
) -> dict:
    """Convert one official surface-point pose to the Ours display schema."""

    point = np.asarray(point_m, dtype=float)
    rotation = np.asarray(rotation_official, dtype=float)
    transform = np.asarray(object_world, dtype=float)
    if point.shape != (3,) or rotation.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("point, rotation, and object_world shapes are invalid")
    center_world_mm = (point + rotation[:, 0] * float(depth_m)) * 1000.0
    rotation_ours_world = rotation @ OURS_TO_OFFICIAL_AXES.T
    world_rotation = transform[:3, :3]
    center_object_mm = world_rotation.T @ (center_world_mm - transform[:3, 3])
    rotation_object = world_rotation.T @ rotation_ours_world
    anchor_world_mm = point * 1000.0
    anchor_object_mm = world_rotation.T @ (anchor_world_mm - transform[:3, 3])
    return {
        "translation": center_object_mm.tolist(),
        "rotation_matrix": rotation_object.tolist(),
        "opening_mm": float(width_m) * 1000.0,
        "grasp_width_mm": float(width_m) * 1000.0,
        "depth_mm": float(depth_m) * 1000.0,
        "score_total": float(score_total),
        "anchor_point": anchor_object_mm.tolist(),
        "anchor_normal": rotation_object[:, 2].tolist(),
        "approach_direction": rotation_object[:, 2].tolist(),
        "view_direction": rotation_object[:, 2].tolist(),
        "view_id": 0,
        "anchor_id": 0,
        "approach_id": 0,
    }


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _top_ours(ours_results: Path, common_eval: Path, topk: int) -> list[dict]:
    records = _load_json(ours_results / "grasps.json")
    common = _load_json(common_eval / "common_records.json")
    ranked = [row for row in common if row.get("scored") and float(row.get("mu_min", -1.0)) >= 0.0]
    ranked.sort(key=lambda row: float(row["mu_min"]))
    return [records[int(row["index"])] for row in ranked[: int(topk)]]


def _top_gn(merged_run: Path, topk: int, object_world: np.ndarray) -> list[dict]:
    labels = np.load(merged_run / "labels.npz")
    points = np.asarray(labels["points"], dtype=np.float32)
    offsets = np.asarray(labels["offsets"], dtype=np.float32)
    scores = np.asarray(labels["scores"], dtype=np.float32)
    valid = np.flatnonzero(np.isfinite(scores.reshape(-1)) & (scores.reshape(-1) >= 0.0))
    valid = valid[np.argsort(scores.reshape(-1)[valid], kind="stable")[: int(topk)]]
    config = DenseAnnotationConfig.full()
    views = generate_views(config.num_views)
    rotations = generate_view_rotations(
        np.repeat(views, config.num_angles, axis=0),
        np.tile(np.arange(config.num_angles, dtype=np.float32) * (np.pi / config.num_angles), config.num_views),
    ).reshape(config.num_views, config.num_angles, 3, 3)
    records = []
    for flat_id in valid:
        point_id, view_id, angle_id, depth_id = np.unravel_index(int(flat_id), scores.shape)
        records.append(
            official_candidate_to_record(
                points[point_id], rotations[view_id, angle_id], config.depths_m[depth_id],
                offsets[point_id, view_id, angle_id, depth_id, 2], object_world,
                score_total=float(scores.reshape(-1)[flat_id]),
            )
        )
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--ours-results", type=Path, required=True)
    parser.add_argument("--ours-common", type=Path, required=True)
    parser.add_argument("--gn-run", type=Path, required=True, help="merged GN labels.npz directory")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--save-image", type=Path)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args(argv)
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    cloud, object_data = make_object_cloud(object_loader_argument(args.object))
    ours = _top_ours(args.ours_results, args.ours_common, args.topk)
    gn = _top_gn(args.gn_run, args.topk, np.asarray(object_data.T_object_world, dtype=float))
    geometries = [cloud]
    for record in ours:
        geometries.append(make_gripper_lineset(record, color=(0.9, 0.1, 0.1), object_data=object_data))
    for record in gn:
        geometries.append(make_gripper_lineset(record, color=(0.1, 0.2, 0.9), object_data=object_data))
    print({"ours_topk": len(ours), "gn_topk": len(gn), "ours_color": "red", "gn_color": "blue"})
    show_geometries(
        geometries,
        title=f"Common evaluator Top-{args.topk}: Ours(red) vs GN(blue)",
        save_image=str(args.save_image) if args.save_image else None,
        show_window=not args.no_window,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
