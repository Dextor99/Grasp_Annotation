"""Evaluate frozen Ours poses against the official point-cloud geometry test.

This adapter deliberately does not load an OBJ/SDF or run Dex-Net force
closure.  Both methods can therefore be compared on the exact same source
PLY surface cloud.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.graspnet_annotation.official_adapter import evaluate_official_collision
from baselines.graspnet_annotation.preprocess import load_surface_ply_in_metres
from scripts.common_eval.ours_official_common_eval import record_to_official_components


def evaluate_ours_ply_geometry(records: list[dict], model_points_m: np.ndarray, *,
                               outlier_m: float = 0.05, empty_thresh: int = 10,
                               T_object_world: np.ndarray | None = None,
                               native_raw_count: int | None = None,
                               native_unique_count: int | None = None) -> tuple[dict, list[dict]]:
    """Run official collision/empty checks and return auditable geometry rows."""
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    points = np.asarray(model_points_m, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("model_points_m must be a non-empty finite (N, 3) array")
    components = [record_to_official_components(record, T_object_world) for record in records]
    rows = np.stack([component["grasp_row"] for component in components], axis=0) if components else np.empty((0, 17), dtype=np.float32)
    started = time.perf_counter()
    if len(records):
        collision, empty = evaluate_official_collision(
            rows, points, points, outlier_m=outlier_m, empty_thresh=empty_thresh,
        )
    else:
        collision = np.empty(0, dtype=bool)
        empty = np.empty(0, dtype=bool)
    collision = np.asarray(collision, dtype=bool).reshape(-1)
    empty = np.asarray(empty, dtype=bool).reshape(-1)
    if len(collision) != len(records) or len(empty) != len(records):
        raise RuntimeError("official collision evaluator returned an unexpected result length")
    geometry_valid = ~collision & ~empty
    elapsed = time.perf_counter() - started
    raw_count = int(len(records) if native_raw_count is None else native_raw_count)
    unique_count = int(len(records) if native_unique_count is None else native_unique_count)
    if raw_count < unique_count or unique_count != len(records):
        raise ValueError("native raw/unique counts must satisfy raw >= unique == evaluated record count")
    valid_count = int(geometry_valid.sum())
    summary = {
        "evaluation_mode": "ply_common_geometry",
        "n_candidates": raw_count,
        "n_raw_candidates": raw_count,
        "n_unique_outputs": unique_count,
        "n_common_geometry_valid": valid_count,
        "common_geometry_valid": valid_count,
        "common_geometry_valid_rate_output": float(valid_count / unique_count) if unique_count else 0.0,
        "common_geometry_yield_raw": float(valid_count / raw_count) if raw_count else 0.0,
        "common_geometry_eval_time_s": float(elapsed),
        "n_collision": int(collision.sum()),
        "n_empty": int(empty.sum()),
        "native_score_order_preserved": True,
    }
    output_rows = [
        {
            "index": int(i), "collision": bool(collision[i]), "empty": bool(empty[i]),
            "geometry_valid": bool(geometry_valid[i]),
        }
        for i in range(len(records))
    ]
    return summary, output_rows


def _write_outputs(output: Path, summary: dict, rows: list[dict], records: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "common_records.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "common_eval.npz",
        candidate_ids=np.arange(len(rows), dtype=np.int64),
        collision=np.asarray([r["collision"] for r in rows], dtype=bool),
        empty=np.asarray([r["empty"] for r in rows], dtype=bool),
        geometry_valid=np.asarray([r["geometry_valid"] for r in rows], dtype=bool),
    )
    (output / "ours_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-results", type=Path, required=True)
    parser.add_argument("--ours-object", type=Path, required=True, help="same source PLY used by Ours")
    parser.add_argument("--input-unit", choices=("m", "mm", "cm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--surface-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--empty-thresh", type=int, default=10)
    parser.add_argument("--outlier-m", type=float, default=0.05)
    args = parser.parse_args(argv)
    records_path = args.ours_results / "grasps.json"
    if not records_path.is_file():
        raise FileNotFoundError(f"missing Ours records: {records_path}")
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if not args.ours_object.is_file():
        candidate = PROJECT_ROOT / args.ours_object
        if candidate.is_file():
            args.ours_object = candidate
    if not args.ours_object.is_file():
        raise FileNotFoundError(f"source PLY not found: {args.ours_object}")
    meta_path = args.ours_results / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    from object_preprocess import prepare_object

    transform = np.asarray(prepare_object(str(args.ours_object)).T_object_world, dtype=float)
    points = load_surface_ply_in_metres(args.ours_object, args.input_unit)
    # Keep the CLI's sampling/seed parameters explicit in metadata while the
    # common geometry protocol evaluates the complete original PLY cloud.
    summary, rows = evaluate_ours_ply_geometry(
        records, points, empty_thresh=args.empty_thresh, outlier_m=args.outlier_m,
        T_object_world=transform,
        native_raw_count=int(meta.get("raw_grasp_count", len(records))),
        native_unique_count=int(meta.get("unique_grasp_count", len(records))),
    )
    summary.update({
        "source_ours_results": str(args.ours_results), "ours_object": str(args.ours_object),
        "model_point_count": int(len(points)), "input_unit": args.input_unit,
        "surface_samples": int(args.surface_samples), "seed": int(args.seed),
        "native_generation_runtime_s": float(
            meta.get("timings", {}).get("total_s", 0.0)
            if isinstance(meta.get("timings", {}), dict) else 0.0
        ),
        "evaluator": "official_graspnet_collision_empty_return_dexgrasps_false",
    })
    _write_outputs(args.output, summary, rows, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
