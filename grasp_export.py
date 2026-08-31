"""Three-file export for finalized grasp annotations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _array(records, field, shape, dtype=float):
    if not records:
        return np.empty((0, *shape), dtype=dtype)
    return np.asarray([record[field] for record in records], dtype=dtype).reshape(
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
        "depths_mm": _array(records, "depth_mm", (), dtype=float),
        "scores_total": _array(records, "score_total", (), dtype=float),
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
