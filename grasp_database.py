"""Persistence for portable multi-view 6D grasp annotations."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class GraspDatasetPaths:
    json_path: Path
    npz_path: Path
    meta_path: Path


def _valid_pose(grasp):
    """Return a finite rigid transform from a grasp record, or ``None``."""
    try:
        pose = np.asarray(grasp["T_gripper_object"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        return None
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        return None
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        return None
    return pose


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value):
    """Convert JSON-safe NumPy containers without accepting opaque objects."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        converted = [_json_safe(item) for item in value]
        return converted if all(item is not _UNSAFE for item in converted) else _UNSAFE
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return _UNSAFE
            safe_item = _json_safe(item)
            if safe_item is _UNSAFE:
                return _UNSAFE
            converted[key] = safe_item
        return converted
    return _UNSAFE


_UNSAFE = object()
_CORE_FIELDS = {
    "T_gripper_object", "id", "translation", "rotation", "quaternion_xyzw",
    "opening", "gripper_width", "score_total", "view_id", "view_direction",
}


def _normalised_direction(value):
    try:
        direction = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        return None
    length = np.linalg.norm(direction)
    return None if length <= 0.0 else (direction / length).tolist()


def _record(grasp, index, pose):
    rotation = pose[:3, :3]
    opening = _finite_number(grasp.get("opening", grasp.get("gripper_width")))
    total = _finite_number(grasp.get("score_total"))
    record = {
        "id": f"grasp-{index:06d}",
        "translation": pose[:3, 3].tolist(),
        "rotation": rotation.reshape(-1).tolist(),
        "quaternion_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
        "opening": opening,
        "gripper_width": opening,
        "score_total": total,
        "view_id": _json_safe(grasp.get("view_id")),
        "view_direction": _normalised_direction(grasp.get("view_direction")),
    }
    for key, value in grasp.items():
        if key in _CORE_FIELDS:
            continue
        if key.startswith("score_"):
            score = _finite_number(value)
            if score is not None:
                record[key] = score
            continue
        safe_value = _json_safe(value)
        if safe_value is not _UNSAFE:
            record[key] = safe_value
    return record


def save_grasp_dataset(grasps, output_directory, metadata):
    """Write validated grasp annotations as JSON, NPZ, and metadata JSON files."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    poses = []
    for grasp in grasps:
        if not isinstance(grasp, dict):
            continue
        pose = _valid_pose(grasp)
        if pose is None:
            continue
        records.append(_record(grasp, len(records), pose))
        poses.append(pose)

    pose_array = np.asarray(poses, dtype=float).reshape((-1, 4, 4))
    translations = pose_array[:, :3, 3]
    rotations = pose_array[:, :3, :3]
    quaternions = np.asarray([record["quaternion_xyzw"] for record in records], dtype=float).reshape((-1, 4))
    openings = np.asarray([record["opening"] if record["opening"] is not None else np.nan for record in records], dtype=float)
    scores = np.asarray([record["score_total"] if record["score_total"] is not None else np.nan for record in records], dtype=float)
    view_ids = np.asarray([record["view_id"] if isinstance(record["view_id"], (int, float)) else -1 for record in records], dtype=int)

    paths = GraspDatasetPaths(
        output_directory / "grasps.json", output_directory / "grasps.npz", output_directory / "meta.json"
    )
    paths.json_path.write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
    np.savez(
        paths.npz_path, poses=pose_array, translations=translations, rotations=rotations,
        quaternions=quaternions, openings=openings, scores=scores, view_ids=view_ids,
    )
    metadata = {} if metadata is None else dict(metadata)
    safe_metadata = _json_safe(metadata)
    if safe_metadata is _UNSAFE:
        raise ValueError("metadata must be JSON serializable")
    safe_metadata.setdefault("units", "mm")
    safe_metadata.setdefault("visibility_strategy", "normal_based_front_facing")
    paths.meta_path.write_text(json.dumps(safe_metadata, indent=2, allow_nan=False), encoding="utf-8")
    return paths
