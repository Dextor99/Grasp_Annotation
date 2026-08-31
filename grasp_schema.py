"""Serialization-safe final schema for scored grasp annotations."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy.spatial.transform import Rotation


SCORE_FIELDS = (
    "score_force_closure",
    "score_inner_points_ratio",
    "score_zmin",
    "score_zmax",
    "score_proj_area",
    "score_proj_area_ratio",
    "score_y_diff",
    "score_y0_diff",
    "score_angle_ymin",
    "score_angle_ymax",
    "score_angle_diff",
)


def _finite_float(value, field):
    if not isinstance(value, Real) or not np.isfinite(float(value)):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _finite_vector(grasp, field):
    if field not in grasp:
        raise ValueError(f"missing required grasp field: {field}")
    vector = np.asarray(grasp[field], dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{field} must be a finite 3-vector")
    return vector.tolist()


def _pose(grasp):
    transform = np.asarray(grasp.get("T_gripper_object"), dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_gripper_object must be a finite 4x4 matrix")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("T_gripper_object rotation must be a valid SO(3) matrix")
    return transform, rotation


def _source_ids(grasp):
    records = []
    seen = set()
    for source in grasp.get("source_ids", []):
        triple = tuple(
            _source_int(source[field], field)
            for field in ("view_id", "anchor_id", "approach_id")
        )
        if triple not in seen:
            seen.add(triple)
            records.append(
                {"view_id": triple[0], "anchor_id": triple[1], "approach_id": triple[2]}
            )
    direct = tuple(
        _source_int(grasp[field], field)
        for field in ("view_id", "anchor_id", "approach_id")
    )
    if all(value is not None for value in direct) and direct not in seen:
        records.append(
            {"view_id": direct[0], "anchor_id": direct[1], "approach_id": direct[2]}
        )
    return sorted(records, key=lambda item: (item["view_id"], item["anchor_id"], item["approach_id"]))


def _source_set(grasp, plural, singular):
    values = {_source_int(value, plural) for value in grasp.get(plural, [])}
    if grasp.get(singular) is not None:
        values.add(_source_int(grasp[singular], singular))
    return sorted(values)


def _source_int(value, field):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must contain integer identifiers")
    return int(value)


def normalize_grasp_record(grasp):
    """Convert an internal candidate to the fixed, JSON-safe final record."""
    transform, rotation = _pose(grasp)
    for field in ("view_id", "anchor_id", "approach_id"):
        if field not in grasp:
            raise ValueError(f"missing required grasp field: {field}")

    search_opening_mm = _finite_float(
        grasp.get("search_opening_mm", grasp.get("opening")),
        "search_opening_mm",
    )
    grasp_width_mm = _finite_float(
        grasp.get("grasp_width_mm", grasp.get("opening")),
        "grasp_width_mm",
    )
    support_span_mm = _finite_float(
        grasp.get("support_span_mm", grasp.get("score_y_diff", 0.0)),
        "support_span_mm",
    )
    if grasp_width_mm > search_opening_mm + 1e-9:
        raise ValueError("grasp_width_mm must not exceed search_opening_mm")
    if support_span_mm > search_opening_mm + 1e-9:
        raise ValueError("support_span_mm must not exceed search_opening_mm")

    record = {
        "translation": transform[:3, 3].astype(float).tolist(),
        "rotation_matrix": rotation.astype(float).tolist(),
        "quaternion_xyzw": Rotation.from_matrix(rotation).as_quat().astype(float).tolist(),
        "opening_mm": _finite_float(grasp.get("opening"), "opening"),
        "search_opening_mm": search_opening_mm,
        "grasp_width_mm": grasp_width_mm,
        "support_span_mm": support_span_mm,
        "closure_center_offset_mm": _finite_float(
            grasp.get("closure_center_offset_mm", 0.0),
            "closure_center_offset_mm",
        ),
        "closure_margin_mm": _finite_float(
            grasp.get("closure_margin_mm", 0.0),
            "closure_margin_mm",
        ),
        "requested_margin_mm": _finite_float(
            grasp.get("requested_margin_mm", grasp.get("closure_margin_mm", 0.0)),
            "requested_margin_mm",
        ),
        "effective_margin_mm": _finite_float(
            grasp.get("effective_margin_mm", grasp.get("closure_margin_mm", 0.0)),
            "effective_margin_mm",
        ),
        "closure_refined": bool(grasp.get("closure_refined", False)),
        "closure_geometry_valid": bool(grasp.get("closure_geometry_valid", True)),
        "closure_pose_valid": bool(grasp.get("closure_pose_valid", True)),
        "depth_mm": _finite_float(grasp.get("depth"), "depth"),
        "score_total": _finite_float(grasp.get("score_total"), "score_total"),
        "view_id": _source_int(grasp["view_id"], "view_id"),
        "anchor_id": _source_int(grasp["anchor_id"], "anchor_id"),
        "approach_id": _source_int(grasp["approach_id"], "approach_id"),
        "view_direction": _finite_vector(grasp, "view_direction"),
        "anchor_point": _finite_vector(grasp, "anchor_point"),
        "anchor_normal": _finite_vector(grasp, "anchor_normal"),
        "approach_direction": _finite_vector(grasp, "approach_direction"),
        "source_view_ids": _source_set(grasp, "source_view_ids", "view_id"),
        "source_anchor_ids": _source_set(grasp, "source_anchor_ids", "anchor_id"),
        "source_approach_ids": _source_set(grasp, "source_approach_ids", "approach_id"),
        "source_ids": _source_ids(grasp),
    }
    if "approach_offset_deg" in grasp:
        record["approach_offset_deg"] = _finite_float(
            grasp["approach_offset_deg"], "approach_offset_deg"
        )
    for field in SCORE_FIELDS:
        value = grasp.get(field)
        record[field] = (
            float(value)
            if isinstance(value, Real) and np.isfinite(float(value))
            else None
        )
    before_y0 = grasp.get("score_y0_diff_before_refinement", grasp.get("score_y0_diff"))
    record["score_y0_diff_before_refinement"] = (
        float(before_y0)
        if isinstance(before_y0, Real) and np.isfinite(float(before_y0))
        else None
    )
    refined_y0 = grasp.get("score_y0_diff_refined")
    record["score_y0_diff_refined"] = (
        float(refined_y0)
        if isinstance(refined_y0, Real) and np.isfinite(float(refined_y0))
        else None
    )
    return record
