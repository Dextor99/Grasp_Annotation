"""Prepare a non-destructive, metre-scale OBJ for GN-Full SDF evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from baselines.graspnet_annotation.preprocess import UNIT_TO_METRES


def prepare(source: Path, output_dir: Path, input_unit: str) -> dict:
    if input_unit not in UNIT_TO_METRES:
        raise ValueError(f"unknown input unit: {input_unit}")
    if not source.is_file():
        raise FileNotFoundError(source)
    loaded = trimesh.load_mesh(source, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"source has no triangle mesh: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    scaled = loaded.copy()
    scaled.vertices = np.asarray(scaled.vertices, dtype=np.float64) * UNIT_TO_METRES[input_unit]
    scaled_path = output_dir / f"{source.stem}_scaled.obj"
    scaled.export(scaled_path)

    repaired = scaled.copy()
    repaired.remove_degenerate_faces()
    repaired.remove_duplicate_faces()
    trimesh.repair.fix_winding(repaired)
    trimesh.repair.fix_inversion(repaired)
    trimesh.repair.fill_holes(repaired)
    repair_method = "in_place_repair"
    if not repaired.is_watertight:
        # Keep this explicitly marked as an evaluation fallback.  The source
        # mesh remains untouched and is always preserved in the metadata.
        repaired = repaired.convex_hull
        repair_method = "convex_hull_fallback"
    repaired_path = output_dir / f"{source.stem}_repaired.obj"
    repaired.export(repaired_path)

    metadata = {
        "source_mesh": str(source),
        "source_unit": input_unit,
        "scale_to_metres": UNIT_TO_METRES[input_unit],
        "source_vertex_count": int(len(loaded.vertices)),
        "source_face_count": int(len(loaded.faces)),
        "source_watertight": bool(loaded.is_watertight),
        "scaled_mesh": scaled_path.name,
        "repaired_mesh": repaired_path.name,
    }
    (output_dir / "original_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report = {
        "method": repair_method,
        "repaired_vertex_count": int(len(repaired.vertices)),
        "repaired_face_count": int(len(repaired.faces)),
        "repaired_watertight": bool(repaired.is_watertight),
        "warning": "Convex-hull fallback changes geometry; use only as an explicitly labelled evaluation asset."
        if repair_method == "convex_hull_fallback" else None,
    }
    (output_dir / "repair_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {**metadata, **report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-unit", choices=tuple(UNIT_TO_METRES), required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output_dir, args.input_unit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
