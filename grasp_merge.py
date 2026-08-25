"""Pose validation and score-ordered grasp deduplication."""

import math

import numpy as np


def rotation_angle_degrees(rotation1, rotation2):
    """Return the angular distance between two rotation matrices in degrees."""
    relative_rotation = np.asarray(rotation1, dtype=float).T @ np.asarray(rotation2, dtype=float)
    cosine = (np.trace(relative_rotation) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _valid_pose(grasp):
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


def _score_for_sorting(grasp):
    try:
        score = float(grasp.get("score_total", float("-inf")))
    except (AttributeError, TypeError, ValueError):
        return float("-inf")
    return score if math.isfinite(score) else float("-inf")


def _is_duplicate(pose1, pose2, position_threshold_mm, rotation_threshold_deg):
    position_distance = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
    return (
        position_distance <= position_threshold_mm
        and rotation_angle_degrees(pose1[:3, :3], pose2[:3, :3]) <= rotation_threshold_deg
    )


def merge_duplicate_grasps(grasps, position_threshold_mm=5.0, rotation_threshold_deg=10.0):
    """Return valid, score-ranked grasps with near-duplicate poses removed."""
    if not np.isfinite(position_threshold_mm) or position_threshold_mm < 0:
        raise ValueError("position_threshold_mm must be finite and non-negative")
    if not np.isfinite(rotation_threshold_deg) or rotation_threshold_deg < 0:
        raise ValueError("rotation_threshold_deg must be finite and non-negative")

    selected = []
    selected_poses = []
    for grasp in sorted(grasps, key=_score_for_sorting, reverse=True):
        pose = _valid_pose(grasp)
        if pose is None:
            continue
        if not any(
            _is_duplicate(pose, selected_pose, position_threshold_mm, rotation_threshold_deg)
            for selected_pose in selected_poses
        ):
            selected.append(grasp)
            selected_poses.append(pose)
    return selected
