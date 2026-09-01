"""Validate and merge force-closure shard outputs into raw labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def merge(geometry_run: Path, shard_dir: Path, output: Path) -> dict:
    labels = np.load(geometry_run / "grasp_labels.npz")
    collision = np.asarray(labels["collision"], dtype=bool)
    valid_ids = np.flatnonzero(~collision.reshape(-1)).astype(np.int64)
    shard_paths = sorted(shard_dir.glob("fc_shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard files found in {shard_dir}")
    ids, scores, scored, errors = [], [], [], []
    for path in shard_paths:
        data = np.load(path)
        shard_ids = np.asarray(data["candidate_ids"], dtype=np.int64).reshape(-1)
        shard_scores = np.asarray(data["mu_min"], dtype=np.float32).reshape(-1)
        shard_scored = np.asarray(data["scored_mask"], dtype=bool).reshape(-1)
        shard_errors = np.asarray(data["error_mask"], dtype=bool).reshape(-1)
        lengths = {len(shard_ids), len(shard_scores), len(shard_scored), len(shard_errors)}
        if len(lengths) != 1:
            raise RuntimeError(f"shard {path.name} has inconsistent array length: "
                               f"candidate_ids={len(shard_ids)}, mu_min={len(shard_scores)}, "
                               f"scored_mask={len(shard_scored)}, error_mask={len(shard_errors)}")
        ids.append(shard_ids)
        scores.append(shard_scores)
        scored.append(shard_scored)
        errors.append(shard_errors)
    merged_ids = np.concatenate(ids) if ids else np.empty(0, dtype=np.int64)
    merged_scores = np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
    merged_scored = np.concatenate(scored) if scored else np.empty(0, dtype=bool)
    merged_errors = np.concatenate(errors) if errors else np.empty(0, dtype=bool)
    unique_ids, frequencies = np.unique(merged_ids, return_counts=True)
    duplicate_ids = unique_ids[frequencies > 1]
    missing_ids = np.setdiff1d(valid_ids, unique_ids)
    extra_ids = np.setdiff1d(unique_ids, valid_ids)
    nonfinite_scored = ~np.isfinite(merged_scores) & merged_scored
    if (len(duplicate_ids) or len(missing_ids) or len(extra_ids) or np.any(merged_errors)
            or not np.all(merged_scored) or np.any(nonfinite_scored)):
        raise RuntimeError(
            f"shard validation failed: duplicates={len(duplicate_ids)}, missing={len(missing_ids)}, "
            f"extra={len(extra_ids)}, errors={int(merged_errors.sum())}, unscored={int((~merged_scored).sum())}, "
            f"nonfinite={int(nonfinite_scored.sum())} (all scored values must be finite)"
        )
    merged = np.asarray(labels["scores"], dtype=np.float32).reshape(-1).copy()
    merged[merged_ids] = merged_scores
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "labels.npz", points=labels["points"], offsets=labels["offsets"], collision=collision, scores=merged.reshape(collision.shape))
    summary = {"geometry_valid_count": int(len(valid_ids)), "scored_count": int(merged_scored.sum()), "n_fc_errors": int(merged_errors.sum()), "duplicate_ids": int(len(duplicate_ids)), "missing_ids": int(len(missing_ids)), "extra_ids": int(len(extra_ids)), "n_fc_valid": int(np.count_nonzero(merged_scores >= 0.0))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(merge(args.geometry_run, args.shard_dir, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
