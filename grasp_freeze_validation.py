"""Acceptance gates for freezing the finalized grasp annotation method."""

from __future__ import annotations

import numpy as np


class AcceptanceFailure(AssertionError):
    """Raised when a result does not satisfy the algorithm freeze gate."""


def _fail(message):
    raise AcceptanceFailure(message)


def _finite_array(record, field, shape):
    try:
        value = np.asarray(record[field], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        _fail(f"invalid {field}: {error}")
    if value.shape != shape or not np.all(np.isfinite(value)):
        _fail(f"{field} must be finite with shape {shape}")
    return value


def _assert_record(record):
    translation = _finite_array(record, "translation", (3,))
    rotation = _finite_array(record, "rotation_matrix", (3, 3))
    quaternion = _finite_array(record, "quaternion_xyzw", (4,))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        _fail("rotation_matrix is not a valid SO(3) matrix")
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6):
        _fail("quaternion_xyzw must have unit norm")
    if not np.all(np.isfinite(translation)):
        _fail("translation contains non-finite values")

    for field in ("view_direction", "anchor_normal", "approach_direction"):
        direction = _finite_array(record, field, (3,))
        if not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-6):
            _fail(f"{field} must have unit norm")
    _finite_array(record, "anchor_point", (3,))

    for field in ("opening_mm", "depth_mm", "score_total"):
        value = record.get(field)
        if not np.isscalar(value) or not np.isfinite(float(value)):
            _fail(f"{field} must be finite")
    search_opening = record.get("search_opening_mm")
    grasp_width = record.get("grasp_width_mm")
    support_span = record.get("support_span_mm")
    if search_opening is not None and grasp_width is not None:
        if float(grasp_width) > float(search_opening) + 1e-9:
            _fail("grasp_width_mm must not exceed search_opening_mm")
    if search_opening is not None and support_span is not None:
        if float(support_span) > float(search_opening) + 1e-9:
            _fail("support_span_mm must not exceed search_opening_mm")
    if record.get("closure_pose_valid") is False:
        _fail("final annotation contains an invalid refined pose")
    for field, value in record.items():
        if field.startswith("score_") and value is not None and not np.isfinite(float(value)):
            _fail(f"{field} contains a non-finite score")


def _assert_score_order(records, label):
    scores = [float(record["score_total"]) for record in records]
    if any(left < right for left, right in zip(scores, scores[1:])):
        _fail(f"{label} are not sorted by descending score_total")


def assert_annotation_invariants(result):
    """Require nonempty, finite, sorted output with a valid merge count."""
    raw_count = len(result.raw_grasps)
    unique_count = len(result.unique_grasps)
    if raw_count <= 0:
        _fail("raw_grasps must be nonempty")
    if unique_count <= 0:
        _fail("unique_grasps must be nonempty")
    if unique_count > raw_count:
        _fail("SE(3) merge cannot increase unique_grasps above raw_grasps")
    reduction_ratio = result.meta.get("merge_reduction_ratio")
    if reduction_ratio is not None:
        if not np.isscalar(reduction_ratio) or not np.isfinite(float(reduction_ratio)):
            _fail("merge_reduction_ratio must be finite")
        expected_ratio = 1.0 - unique_count / raw_count
        if not np.isclose(float(reduction_ratio), expected_ratio, atol=1e-12):
            _fail("merge_reduction_ratio does not match grasp counts")
    if result.meta.get("raw_grasp_count") != raw_count:
        _fail("meta raw_grasp_count does not match records")
    if result.meta.get("unique_grasp_count") != unique_count:
        _fail("meta unique_grasp_count does not match records")

    counts = result.meta.get("candidate_counts")
    if counts is not None:
        expected = {
            "closure_valid_count": raw_count,
            "unique_grasp_count": unique_count,
        }
        for key, value in expected.items():
            if counts.get(key) != value:
                _fail(f"candidate_counts[{key}] does not match records")
        for key in (
            "generated_candidate_count",
            "scored_candidate_count",
            "refinement_input_count",
            "closure_geometry_rejected",
            "closure_pose_collision_rejected",
        ):
            if key not in counts or not isinstance(counts[key], int) or counts[key] < 0:
                _fail(f"candidate_counts[{key}] must be a non-negative integer")
        if counts["closure_geometry_rejected"] + counts["closure_pose_collision_rejected"] > counts["refinement_input_count"]:
            _fail("candidate closure rejection counts exceed refinement input")

    for record in result.raw_grasps:
        _assert_record(record)
    for record in result.unique_grasps:
        _assert_record(record)
    _assert_score_order(result.raw_grasps, "raw_grasps")
    _assert_score_order(result.unique_grasps, "unique_grasps")


def assert_repeated_results_equal(reference, repeated, top_k=10):
    """Require exact deterministic count, Top-K pose, and score equality."""
    if len(reference.raw_grasps) != len(repeated.raw_grasps):
        _fail("raw grasp counts differ between repeated runs")
    if len(reference.unique_grasps) != len(repeated.unique_grasps):
        _fail("unique grasp counts differ between repeated runs")
    count = min(top_k, len(reference.unique_grasps))
    for index in range(count):
        expected = reference.unique_grasps[index]
        actual = repeated.unique_grasps[index]
        for field in ("translation", "rotation_matrix", "quaternion_xyzw"):
            if not np.array_equal(np.asarray(expected[field]), np.asarray(actual[field])):
                _fail(f"Top-{count} {field} differs at rank {index}")
        if expected["score_total"] != actual["score_total"]:
            _fail(f"Top-{count} score_total differs at rank {index}")
