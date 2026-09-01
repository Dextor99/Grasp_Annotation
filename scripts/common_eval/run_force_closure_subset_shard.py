"""Score one deterministic GN-Full subset shard in a fresh process."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from baselines.graspnet_annotation.config import DenseAnnotationConfig
from baselines.graspnet_annotation.official_adapter import (
    build_dexnet_grasp,
    build_force_closure_configs,
    load_dexnet_model,
    score_official_force_closure_prepared,
)
from baselines.graspnet_annotation.view_sampling import generate_view_rotations, generate_views


def run(
    geometry_run: Path,
    sdf_prefix: Path,
    candidate_ids_path: Path,
    start: int,
    end: int,
    output: Path,
) -> dict:
    if start < 0 or end <= start:
        raise ValueError("shard range must satisfy 0 <= start < end")
    labels = np.load(geometry_run / "grasp_labels.npz")
    points = np.asarray(labels["points"], dtype=np.float32)
    collision = np.asarray(labels["collision"], dtype=bool)
    offsets = np.asarray(labels["offsets"], dtype=np.float32)
    requested = np.asarray(np.load(candidate_ids_path), dtype=np.int64).reshape(-1)
    if end > len(requested):
        raise ValueError("shard end exceeds candidate id manifest")
    selected = requested[start:end]
    if len(selected) == 0:
        raise ValueError("subset shard must contain at least one candidate")
    if np.any(selected < 0) or np.any(selected >= collision.size):
        raise ValueError("candidate_ids contain values outside the geometry tensor")
    if np.any(collision.reshape(-1)[selected]):
        raise ValueError("candidate_ids must all be geometry-valid")

    config = DenseAnnotationConfig.full()
    views = generate_views(config.num_views)
    rotations = generate_view_rotations(
        np.repeat(views, config.num_angles, axis=0),
        np.tile(np.arange(config.num_angles, dtype=np.float32) * (np.pi / config.num_angles), config.num_views),
    ).reshape(config.num_views, config.num_angles, 3, 3)
    dex_model = load_dexnet_model(sdf_prefix)
    fc_list, fc_configs = build_force_closure_configs(config.friction_coefficients)
    scores = np.full(len(selected), -1.0, dtype=np.float32)
    scored = np.zeros(len(selected), dtype=bool)
    errors = np.zeros(len(selected), dtype=bool)
    messages = []
    started = time.perf_counter()
    for row, flat_id in enumerate(selected):
        point_id, view_id, angle_id, depth_id = np.unravel_index(int(flat_id), collision.shape)
        depth = float(config.depths_m[depth_id])
        width = float(offsets[point_id, view_id, angle_id, depth_id, 2])
        try:
            grasp = build_dexnet_grasp(points[point_id], rotations[view_id, angle_id], depth, width)
            score = score_official_force_closure_prepared(grasp, dex_model, fc_list, fc_configs)
            if not np.isfinite(score):
                raise ValueError(f"force-closure scorer returned non-finite score: {score!r}")
            scores[row] = score
            scored[row] = True
        except Exception as exc:  # audit every failed candidate instead of silently dropping it
            errors[row] = True
            messages.append({"row": row, "candidate_id": int(flat_id), "error": repr(exc)})
    elapsed = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        candidate_ids=selected,
        mu_min=scores,
        scored_mask=scored,
        error_mask=errors,
        elapsed_s=np.asarray(elapsed, dtype=np.float64),
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {"start": start, "end": end, "count": len(selected), "n_errors": int(errors.sum()), "errors": messages, "elapsed_s": elapsed},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"count": len(selected), "n_errors": int(errors.sum()), "elapsed_s": elapsed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--sdf-prefix", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.geometry_run, args.sdf_prefix, args.candidate_ids, args.start, args.end, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
