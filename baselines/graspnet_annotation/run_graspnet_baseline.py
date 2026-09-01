"""Streamed GN-Full object-level annotation runner."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .candidate_generation import iter_candidate_batches
from .config import DenseAnnotationConfig
from .export import export_annotation_run
from .grasp_point_sampling import sample_grasp_points
from .gripper_geometry import evaluate_adaptive_width
from .label_arrays import RawLabelArrays
from .official_adapter import (
    build_dexnet_grasp,
    build_force_closure_configs,
    load_dexnet_model,
    score_official_force_closure_prepared,
)
from .preprocess import load_mesh_in_metres, validate_mesh_readiness


def run(mesh_path: str | Path, output: str | Path, *, input_unit: str = "m", sdf_prefix: str | Path | None = None,
        max_points: int | None = None, skip_force_closure: bool = False,
        max_force_closure_candidates: int | None = None) -> dict:
    if max_force_closure_candidates is not None and int(max_force_closure_candidates) <= 0:
        raise ValueError("max_force_closure_candidates must be positive")
    config = DenseAnnotationConfig.full(input_unit=input_unit)
    started = time.perf_counter()
    sdf_file = None
    if sdf_prefix is not None:
        sdf_path = Path(sdf_prefix)
        sdf_file = sdf_path if sdf_path.suffix.lower() == ".sdf" else sdf_path.with_suffix(".sdf")
    readiness = validate_mesh_readiness(mesh_path, sdf_file, require_sdf=not skip_force_closure)
    loaded = load_mesh_in_metres(mesh_path, input_unit)
    grasp_points = sample_grasp_points(loaded.mesh, config, max_points=max_points)
    scene_points = sample_grasp_points(loaded.mesh, config, max_points=config.surface_samples)
    dex_model = None if skip_force_closure else load_dexnet_model(Path(sdf_prefix).with_suffix("") if Path(sdf_prefix).suffix.lower() == ".sdf" else sdf_prefix)
    fc_list, fc_configs = (None, None) if skip_force_closure else build_force_closure_configs(config.friction_coefficients)

    labels = RawLabelArrays.create(len(grasp_points), config)
    labels.points[:] = grasp_points
    counts = {"raw_candidates": 0, "empty_count": 0, "collision_count": 0, "geometry_valid_count": 0, "force_closure_valid": 0}
    force_closure_evaluated = 0
    stage = {"preprocess": time.perf_counter() - started, "candidate_generation": 0.0, "width_collision": 0.0, "force_closure": 0.0}
    for point_id in range(len(grasp_points)):
        point_started = time.perf_counter()
        batch = next(iter_candidate_batches(grasp_points[point_id:point_id + 1], config, point_batch_size=1))
        stage["candidate_generation"] += time.perf_counter() - point_started
        widths = np.zeros(batch.size, dtype=np.float32)
        collision = np.ones(batch.size, dtype=bool)
        scores = np.full(batch.size, -1.0, dtype=np.float32)
        dexgrasps = [None] * batch.size
        width_started = time.perf_counter()
        for index in range(batch.size):
            depth = float(config.depths_m[int(batch.depth_ids[index])])
            result = evaluate_adaptive_width(
                scene_points, grasp_points[point_id], batch.rotations[index], depth,
                max_width_m=config.max_width_m, height_m=config.height_m,
                depth_base_m=config.depth_base_m, finger_width_m=config.finger_width_m,
                bottom_thickness_m=config.bottom_thickness_m, empty_thresh=config.empty_thresh,
                hole_size_m=config.hole_size_m, loose_factor_m=config.width_loose_factor_m,
            )
            widths[index] = result.width_m
            if result.empty:
                counts["empty_count"] += 1
                continue
            if result.collision:
                counts["collision_count"] += 1
                continue
            counts["geometry_valid_count"] += 1
            collision[index] = False
            if not skip_force_closure:
                dexgrasps[index] = build_dexnet_grasp(grasp_points[point_id], batch.rotations[index], depth, result.width_m)
        stage["width_collision"] += time.perf_counter() - width_started
        if not skip_force_closure:
            score_started = time.perf_counter()
            for index, dex_grasp in enumerate(dexgrasps):
                if dex_grasp is None:
                    continue
                if max_force_closure_candidates is not None and force_closure_evaluated >= int(max_force_closure_candidates):
                    break
                try:
                    score = score_official_force_closure_prepared(dex_grasp, dex_model, fc_list, fc_configs)
                except Exception:
                    score = -1.0
                scores[index] = score
                force_closure_evaluated += 1
                if score >= 0.0:
                    counts["force_closure_valid"] += 1
            stage["force_closure"] += time.perf_counter() - score_started
        labels.write_point_result(point_id, batch, widths, collision, scores)
        counts["raw_candidates"] += batch.size

    stage["export"] = 0.0
    export_started = time.perf_counter()
    summary = {
        "mesh": str(mesh_path), "input_unit": input_unit, "internal_unit": "m",
        "mesh_watertight": readiness.is_watertight, "n_input_points": int(len(scene_points)),
        "n_grasp_points": int(len(grasp_points)), "n_candidates": counts["raw_candidates"],
        "n_empty": counts["empty_count"], "n_collision": counts["collision_count"],
        "n_geometry_valid": counts["geometry_valid_count"], "n_fc_valid": counts["force_closure_valid"],
        "valid_rate": counts["force_closure_valid"] / max(1, counts["raw_candidates"]),
        "force_closure_skipped": bool(skip_force_closure), "config": config.to_dict(),
        "force_closure_limit": max_force_closure_candidates,
        "force_closure_evaluated": int(force_closure_evaluated),
        "force_closure_truncated": bool(max_force_closure_candidates is not None and force_closure_evaluated < counts["geometry_valid_count"]),
    }
    stage["export"] = time.perf_counter() - export_started
    stage["total"] = time.perf_counter() - started
    export_annotation_run(output, labels, summary, [{"stage": key, "seconds": value} for key, value in stage.items()], config=config)
    return {**summary, "timing": stage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--input-unit", choices=("m", "mm", "cm"), required=True)
    parser.add_argument("--sdf-prefix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--skip-force-closure", action="store_true")
    parser.add_argument("--max-force-closure-candidates", type=int,
                        help="debug limit; leaves unscored candidates at -1 and marks the run truncated")
    args = parser.parse_args()
    if not args.skip_force_closure and args.sdf_prefix is None:
        parser.error("--sdf-prefix is required unless --skip-force-closure is set")
    result = run(args.mesh, args.output, input_unit=args.input_unit, sdf_prefix=args.sdf_prefix,
                 max_points=args.max_points, skip_force_closure=args.skip_force_closure,
                 max_force_closure_candidates=args.max_force_closure_candidates)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
