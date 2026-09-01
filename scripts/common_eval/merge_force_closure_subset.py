"""Strictly merge force-closure shards for a fixed candidate-id manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def merge_subset(geometry_run: Path, shard_dir: Path, candidate_ids_path: Path, output: Path) -> dict:
    labels = np.load(geometry_run / "grasp_labels.npz")
    collision = np.asarray(labels["collision"], dtype=bool)
    expected = np.asarray(np.load(candidate_ids_path), dtype=np.int64).reshape(-1)
    if len(expected) == 0 or len(np.unique(expected)) != len(expected):
        raise ValueError("candidate id manifest must be non-empty and duplicate-free")
    if np.any(expected < 0) or np.any(expected >= collision.size):
        raise ValueError("candidate id manifest contains out-of-range values")
    if np.any(collision.reshape(-1)[expected]):
        raise ValueError("candidate id manifest contains geometry-colliding candidates")
    shard_paths = sorted(shard_dir.glob("subset_fc_shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no subset shard files found in {shard_dir}")
    chunks = []
    for path in shard_paths:
        data = np.load(path)
        ids = np.asarray(data["candidate_ids"], dtype=np.int64).reshape(-1)
        scores = np.asarray(data["mu_min"], dtype=np.float32).reshape(-1)
        scored = np.asarray(data["scored_mask"], dtype=bool).reshape(-1)
        errors = np.asarray(data["error_mask"], dtype=bool).reshape(-1)
        lengths = {len(ids), len(scores), len(scored), len(errors)}
        if len(lengths) != 1:
            raise RuntimeError(f"shard {path.name} has inconsistent array length")
        chunks.append((ids, scores, scored, errors))
    ids = np.concatenate([item[0] for item in chunks])
    scores = np.concatenate([item[1] for item in chunks])
    scored = np.concatenate([item[2] for item in chunks])
    errors = np.concatenate([item[3] for item in chunks])
    unique, frequencies = np.unique(ids, return_counts=True)
    duplicate = unique[frequencies > 1]
    missing = np.setdiff1d(expected, unique)
    extra = np.setdiff1d(unique, expected)
    nonfinite = scored & ~np.isfinite(scores)
    if len(duplicate) or len(missing) or len(extra) or np.any(errors) or not np.all(scored) or np.any(nonfinite):
        raise RuntimeError(
            "subset shard validation failed: "
            f"duplicates={len(duplicate)}, missing={len(missing)}, extra={len(extra)}, "
            f"errors={int(errors.sum())}, unscored={int((~scored).sum())}, nonfinite={int(nonfinite.sum())}"
        )
    output.mkdir(parents=True, exist_ok=False)
    merged_scores = np.full(collision.size, -1.0, dtype=np.float32)
    merged_scores[ids] = scores
    scored_full = np.zeros(collision.size, dtype=bool)
    error_full = np.zeros(collision.size, dtype=bool)
    scored_full[ids] = scored
    error_full[ids] = errors
    np.savez_compressed(
        output / "labels.npz",
        points=np.asarray(labels["points"], dtype=np.float32),
        offsets=np.asarray(labels["offsets"], dtype=np.float32),
        collision=collision,
        scores=merged_scores.reshape(collision.shape),
    )
    np.save(output / "candidate_ids.npy", expected, allow_pickle=False)
    np.save(output / "scored_mask.npy", scored_full.reshape(collision.shape), allow_pickle=False)
    np.save(output / "error_mask.npy", error_full.reshape(collision.shape), allow_pickle=False)
    summary = {
        "subset_count": int(len(expected)),
        "scored_count": int(scored.sum()),
        "n_fc_errors": int(errors.sum()),
        "duplicate_ids": int(len(duplicate)),
        "missing_ids": int(len(missing)),
        "extra_ids": int(len(extra)),
        "n_fc_valid": int(np.count_nonzero(scores >= 0.0)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(merge_subset(args.geometry_run, args.shard_dir, args.candidate_ids, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
