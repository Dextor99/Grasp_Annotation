"""Inventory paired reference assets for the frozen GN/Ours protocol.

This tool deliberately does not reconstruct a mesh, apply a convex hull, or
run SDF generation.  It records what is available so an operator can provide
or approve a watertight reference before any comparison is run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def classify_asset(*, source_exists: bool, watertight: bool, sdf_exists: bool) -> str:
    if source_exists and watertight and sdf_exists:
        return "A_ready"
    if source_exists and watertight:
        return "A_mesh_needs_sdf"
    if source_exists:
        return "B_repair_and_sdf"
    return "C_reference_required"


def _resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _mesh_info(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "source_exists": False, "source_kind": "missing", "watertight": False,
            "vertices": 0, "faces": 0, "bbox_min": [], "bbox_max": [], "extents": [],
        }
    try:
        import numpy as np
        import trimesh
        loaded = trimesh.load_mesh(path, process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(tuple(loaded.geometry.values())) if loaded.geometry else None
        vertices = getattr(loaded, "vertices", None)
        faces = getattr(loaded, "faces", None)
        if vertices is None:
            return {"source_exists": True, "source_kind": type(loaded).__name__, "watertight": False,
                    "vertices": 0, "faces": 0, "bbox_min": [], "bbox_max": [], "extents": []}
        values = np.asarray(vertices, dtype=float)
        bbox_min = values.min(axis=0).tolist() if len(values) else []
        bbox_max = values.max(axis=0).tolist() if len(values) else []
        return {
            "source_exists": True,
            "source_kind": "triangle_mesh" if faces is not None and len(faces) else "point_cloud",
            "watertight": bool(getattr(loaded, "is_watertight", False)),
            "vertices": int(len(values)), "faces": int(len(faces)) if faces is not None else 0,
            "bbox_min": bbox_min, "bbox_max": bbox_max,
            "extents": (np.asarray(bbox_max) - np.asarray(bbox_min)).tolist() if bbox_min else [],
        }
    except Exception as exc:
        return {
            "source_exists": True, "source_kind": "unreadable", "watertight": False,
            "vertices": 0, "faces": 0, "bbox_min": [], "bbox_max": [], "extents": [],
            "read_error": repr(exc),
        }


def inventory(manifest: dict[str, Any], *, root: Path) -> list[dict[str, Any]]:
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("manifest must contain a non-empty objects list")
    rows: list[dict[str, Any]] = []
    for entry in objects:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError("each object entry requires a name")
        source_value = entry.get("reference_mesh") or entry.get("source_mesh")
        source = _resolve(root, source_value)
        info = _mesh_info(source)
        sdf = _resolve(root, entry.get("sdf"))
        sdf_exists = bool(sdf and sdf.is_file())
        classification = classify_asset(
            source_exists=bool(info["source_exists"]),
            watertight=bool(info["watertight"]),
            sdf_exists=sdf_exists,
        )
        action = {
            "A_ready": "run_asset_gate",
            "A_mesh_needs_sdf": "generate_sdf_then_run_asset_gate",
            "B_repair_and_sdf": "create_minimal_repaired_reference_then_generate_sdf",
            "C_reference_required": "provide_original_mesh_or_explicit_evaluation_reference",
        }[classification]
        rows.append({
            "object": str(entry["name"]),
            "ours_ply": entry.get("ours_ply", ""),
            "source_mesh": str(source) if source else "",
            "source_unit": entry.get("input_unit", ""),
            "sdf": str(sdf) if sdf else "",
            "sdf_exists": sdf_exists,
            "classification": classification,
            "required_action": action,
            **info,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON object manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = inventory(manifest, root=Path.cwd())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "asset_inventory.json"
    csv_path = args.output_dir / "asset_inventory.csv"
    json_path.write_text(json.dumps({"objects": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(rows[0].keys()) if rows else ["object"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"objects": len(rows), "json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

