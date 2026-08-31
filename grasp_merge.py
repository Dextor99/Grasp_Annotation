"""Score-ordered, parallel-jaw-symmetry-aware SE(3) grasp merging."""

from __future__ import annotations

from numbers import Real

import numpy as np


_LOCAL_Z_HALF_TURN = np.diag([-1.0, -1.0, 1.0])


def _validated_pose(grasp):
    transform = np.asarray(grasp.get("T_gripper_object"), dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("each grasp must contain a finite 4x4 T_gripper_object")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("T_gripper_object rotation must be a valid SO(3) matrix")
    return transform


def _validated_score(grasp):
    score = grasp.get("score_total")
    if not isinstance(score, Real) or not np.isfinite(float(score)):
        raise ValueError("each grasp must contain a finite score_total")
    return float(score)


def _rotation_angle_deg(relative_rotation):
    cosine = (np.trace(relative_rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def symmetry_aware_rotation_difference_deg(rotation_a, rotation_b):
    """Smallest rotation error, including the local-z parallel-jaw symmetry."""
    relative = np.asarray(rotation_a, dtype=float).T @ np.asarray(rotation_b, dtype=float)
    return min(
        _rotation_angle_deg(relative),
        _rotation_angle_deg(relative @ _LOCAL_Z_HALF_TURN),
    )


def _existing_ids(grasp, plural_key, singular_key):
    values = set(grasp.get(plural_key, []))
    value = grasp.get(singular_key)
    if value is not None:
        values.add(value)
    return values


def _existing_source_tuples(grasp):
    sources = set()
    for source in grasp.get("source_ids", []):
        if isinstance(source, dict):
            triple = (source.get("view_id"), source.get("anchor_id"), source.get("approach_id"))
        else:
            triple = tuple(source)
        if len(triple) == 3 and all(value is not None for value in triple):
            sources.add(triple)
    direct = (grasp.get("view_id"), grasp.get("anchor_id"), grasp.get("approach_id"))
    if all(value is not None for value in direct):
        sources.add(direct)
    return sources


def _initialize_provenance(grasp):
    result = dict(grasp)
    result["source_view_ids"] = sorted(_existing_ids(grasp, "source_view_ids", "view_id"))
    result["source_anchor_ids"] = sorted(_existing_ids(grasp, "source_anchor_ids", "anchor_id"))
    result["source_approach_ids"] = sorted(
        _existing_ids(grasp, "source_approach_ids", "approach_id")
    )
    result["source_ids"] = [
        {"view_id": view_id, "anchor_id": anchor_id, "approach_id": approach_id}
        for view_id, anchor_id, approach_id in sorted(_existing_source_tuples(grasp))
    ]
    return result


def _merge_provenance(representative, duplicate):
    for plural_key, singular_key in (
        ("source_view_ids", "view_id"),
        ("source_anchor_ids", "anchor_id"),
        ("source_approach_ids", "approach_id"),
    ):
        values = set(representative[plural_key])
        values.update(_existing_ids(duplicate, plural_key, singular_key))
        representative[plural_key] = sorted(values)

    sources = {
        (source["view_id"], source["anchor_id"], source["approach_id"])
        for source in representative["source_ids"]
    }
    sources.update(_existing_source_tuples(duplicate))
    representative["source_ids"] = [
        {"view_id": view_id, "anchor_id": anchor_id, "approach_id": approach_id}
        for view_id, anchor_id, approach_id in sorted(sources)
    ]


def merge_grasp_candidates(
    grasps,
    translation_threshold_mm=5.0,
    rotation_threshold_deg=10.0,
):
    """Greedily merge lower-scored poses close to a higher-scored representative.

    Thresholds are strict, matching the method definition: a pose exactly 5 mm
    or 10 degrees away remains distinct.  This is score-ordered greedy merging,
    not transitive connected-component clustering.
    """
    if translation_threshold_mm <= 0 or rotation_threshold_deg <= 0:
        raise ValueError("merge thresholds must be positive")

    validated = []
    for input_index, grasp in enumerate(grasps):
        transform = _validated_pose(grasp)
        score = _validated_score(grasp)
        validated.append((input_index, score, grasp, transform))
    validated.sort(key=lambda item: (-item[1], item[0]))

    representatives = []
    representative_poses = []
    for _, _, grasp, transform in validated:
        duplicate_index = None
        for index, representative_pose in enumerate(representative_poses):
            translation_distance = np.linalg.norm(
                transform[:3, 3] - representative_pose[:3, 3]
            )
            if not translation_distance < translation_threshold_mm:
                continue
            rotation_distance = symmetry_aware_rotation_difference_deg(
                representative_pose[:3, :3], transform[:3, :3]
            )
            if rotation_distance < rotation_threshold_deg - 1e-9:
                duplicate_index = index
                break

        if duplicate_index is None:
            representatives.append(_initialize_provenance(grasp))
            representative_poses.append(transform)
        else:
            _merge_provenance(representatives[duplicate_index], grasp)

    return representatives
