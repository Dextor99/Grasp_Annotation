"""Three-file export for finalized grasp annotations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _array(records, field, shape, dtype=float, fallback_field=None, default_value=None):
    if not records:
        return np.empty((0, *shape), dtype=dtype)
    values = []
    for record in records:
        if field in record:
            values.append(record[field])
        elif fallback_field is not None and fallback_field in record:
            values.append(record[fallback_field])
        elif default_value is not None:
            values.append(default_value)
        else:
            raise KeyError(field)
    return np.asarray(values, dtype=dtype).reshape(
        len(records), *shape
    )


def export_grasp_annotations(result, output_directory):
    """Write unique grasps to JSON/NPZ and run information to meta JSON."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    allowed_names = {"grasps.json", "grasps.npz", "meta.json"}
    unexpected = sorted(path.name for path in output.iterdir() if path.name not in allowed_names)
    if unexpected:
        raise FileExistsError(
            "output directory contains unrelated files; refusing to overwrite or delete: "
            + ", ".join(unexpected)
        )
    grasps_path = output / "grasps.json"
    arrays_path = output / "grasps.npz"
    meta_path = output / "meta.json"
    records = result.unique_grasps
    grasps_json = json.dumps(
        records, ensure_ascii=False, indent=2, allow_nan=False
    )
    meta_json = json.dumps(
        result.meta, ensure_ascii=False, indent=2, allow_nan=False
    )
    arrays = {
        "translations": _array(records, "translation", (3,)),
        "rotation_matrices": _array(records, "rotation_matrix", (3, 3)),
        "quaternions_xyzw": _array(records, "quaternion_xyzw", (4,)),
        "openings_mm": _array(records, "opening_mm", (), dtype=float),
        "search_openings_mm": _array(
            records, "search_opening_mm", (), dtype=float, fallback_field="opening_mm"
        ),
        "grasp_widths_mm": _array(
            records, "grasp_width_mm", (), dtype=float, fallback_field="opening_mm"
        ),
        "support_spans_mm": _array(
            records, "support_span_mm", (), dtype=float, fallback_field="opening_mm"
        ),
        "closure_center_offsets_mm": _array(
            records, "closure_center_offset_mm", (), dtype=float, default_value=0.0
        ),
        "effective_margins_mm": _array(
            records, "effective_margin_mm", (), dtype=float,
            fallback_field="closure_margin_mm", default_value=0.0
        ),
        "depths_mm": _array(records, "depth_mm", (), dtype=float),
        "scores_total": _array(records, "score_total", (), dtype=float),
        "scores_total_v3": _array(
            records, "score_total_v3", (), dtype=float, fallback_field="score_total"
        ),
        "scores_total_v4": _array(
            records, "score_total_v4", (), dtype=float, fallback_field="score_total"
        ),
        "scores_v4_normal": _array(
            records, "score_v4_normal", (), dtype=float, default_value=0.0
        ),
        "scores_v4_support": _array(
            records, "score_v4_support", (), dtype=float, default_value=0.0
        ),
        "scores_v4_stability": _array(
            records, "score_v4_stability", (), dtype=float, default_value=0.0
        ),
        "view_ids": _array(records, "view_id", (), dtype=np.int64),
        "anchor_ids": _array(records, "anchor_id", (), dtype=np.int64),
        "approach_ids": _array(records, "approach_id", (), dtype=np.int64),
    }

    temporary_grasps = output / ".grasps.json.tmp"
    temporary_arrays = output / ".grasps.npz.tmp"
    temporary_meta = output / ".meta.json.tmp"
    temporary_paths = (temporary_grasps, temporary_arrays, temporary_meta)
    try:
        temporary_grasps.write_text(grasps_json, encoding="utf-8")
        with temporary_arrays.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary_meta.write_text(meta_json, encoding="utf-8")
        temporary_grasps.replace(grasps_path)
        temporary_arrays.replace(arrays_path)
        temporary_meta.replace(meta_path)
    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()
    return {
        "grasps_json": grasps_path,
        "grasps_npz": arrays_path,
        "meta_json": meta_path,
    }
