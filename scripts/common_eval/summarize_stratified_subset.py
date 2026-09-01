"""Aggregate a point-stratified force-closure subset to population estimates.

The full GN geometry pass supplies the population size at every grasp point;
the expensive FC subset supplies scores for a deterministic sample from each
point.  This module keeps the observed (unweighted) audit numbers and adds
candidate-weighted estimates for the full geometry-valid population.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def summarize_stratified_subset(
    *, collision: np.ndarray, candidate_ids: np.ndarray, scores: np.ndarray,
    hq_threshold: float = 0.4,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return population-weighted FC/HQ statistics and per-point audit rows.

    Candidate IDs use the C-order flattening of the GN label tensor.  ``scores``
    may be a full tensor (values outside the subset are normally -1) or a
    vector aligned with ``candidate_ids``.
    """
    collision = np.asarray(collision, dtype=bool)
    if collision.ndim < 2:
        raise ValueError("collision must have a point dimension and candidate dimensions")
    n_points = int(collision.shape[0])
    per_point = int(np.prod(collision.shape[1:]))
    total_candidates = int(collision.size)
    ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("candidate_ids contains duplicates")
    if np.any(ids < 0) or np.any(ids >= total_candidates):
        raise ValueError("candidate_ids contains an out-of-range value")

    score_values = np.asarray(scores, dtype=float)
    if score_values.size == total_candidates:
        selected_scores = score_values.reshape(-1)[ids]
    elif score_values.size == len(ids):
        selected_scores = score_values.reshape(-1)
    else:
        raise ValueError(
            f"scores must contain all {total_candidates} candidates or exactly "
            f"{len(ids)} sampled scores, got {score_values.size}"
        )

    flat_collision = collision.reshape(-1)
    if np.any(flat_collision[ids]):
        raise ValueError("candidate_ids includes geometry-colliding candidates")

    population_sizes = (~collision).reshape(n_points, per_point).sum(axis=1).astype(np.int64)
    total_population = int(population_sizes.sum())
    point_ids = ids // per_point
    rows: list[dict[str, Any]] = []
    weighted_fc_numerator = 0.0
    weighted_hq_numerator = 0.0
    weighted_mu_numerator = 0.0
    weighted_mu_denominator = 0.0
    sample_fc_total = 0
    sample_hq_total = 0
    sample_valid_mu: list[float] = []

    for point_id in range(n_points):
        mask = point_ids == point_id
        point_scores = selected_scores[mask]
        sample_size = int(point_scores.size)
        valid = np.isfinite(point_scores) & (point_scores >= 0.0)
        fc_valid = int(valid.sum())
        hq = valid & (point_scores <= float(hq_threshold))
        hq_count = int(hq.sum())
        valid_mu = point_scores[valid]
        sample_fc_rate = fc_valid / sample_size if sample_size else 0.0
        sample_hq_probability = hq_count / sample_size if sample_size else 0.0
        sample_mean_mu = float(valid_mu.mean()) if valid_mu.size else -1.0
        population_size = int(population_sizes[point_id])
        weight = population_size / total_population if total_population else 0.0
        weighted_fc_numerator += weight * sample_fc_rate
        weighted_hq_numerator += weight * sample_hq_probability
        if valid_mu.size:
            # Weight the per-point mean by its sampled FC fraction; this is
            # equivalent to estimating the population sum of mu values.
            weighted_mu_numerator += weight * sample_fc_rate * sample_mean_mu
            weighted_mu_denominator += weight * sample_fc_rate
        sample_fc_total += fc_valid
        sample_hq_total += hq_count
        sample_valid_mu.extend(float(v) for v in valid_mu)
        rows.append({
            "point_id": point_id,
            "population_size": population_size,
            "sample_size": sample_size,
            "fc_valid": fc_valid,
            "hq_mu04": hq_count,
            "sample_fc_rate": sample_fc_rate,
            "sample_hq_probability": sample_hq_probability,
            "sample_mean_mu": sample_mean_mu,
            "sum_mu": float(valid_mu.sum()) if valid_mu.size else 0.0,
            "weight": weight,
        })

    weighted_fc_rate = weighted_fc_numerator
    weighted_hq_probability = weighted_hq_numerator
    summary: dict[str, Any] = {
        "hq_threshold": float(hq_threshold),
        "raw_candidate_count": total_candidates,
        "population_geometry_valid": total_population,
        "geometry_valid_rate": total_population / total_candidates if total_candidates else 0.0,
        "sample_count": int(len(ids)),
        "sample_fc_valid": int(sample_fc_total),
        "sample_hq_mu04": int(sample_hq_total),
        "unweighted_fc_rate": sample_fc_total / len(ids) if len(ids) else 0.0,
        "unweighted_hq_probability": sample_hq_total / len(ids) if len(ids) else 0.0,
        "unweighted_hq_rate_among_fc": sample_hq_total / sample_fc_total if sample_fc_total else 0.0,
        "unweighted_mean_mu": float(np.mean(sample_valid_mu)) if sample_valid_mu else -1.0,
        "weighted_fc_rate": weighted_fc_rate,
        "weighted_hq_probability": weighted_hq_probability,
        "weighted_hq_rate_among_fc": weighted_hq_probability / weighted_fc_rate if weighted_fc_rate else 0.0,
        "weighted_mean_mu": weighted_mu_numerator / weighted_mu_denominator if weighted_mu_denominator else -1.0,
        "estimated_fc_valid": total_population * weighted_fc_rate,
        "estimated_n_mu_le_04": total_population * weighted_hq_probability,
        "estimated_fc_yield_raw": (total_population * weighted_fc_rate) / total_candidates if total_candidates else 0.0,
        "estimated_hq_yield_raw": (total_population * weighted_hq_probability) / total_candidates if total_candidates else 0.0,
        "estimated_hq_rate_among_fc": weighted_hq_probability / weighted_fc_rate if weighted_fc_rate else 0.0,
        "population_size_sum": total_population,
        "sample_size_sum": int(sum(row["sample_size"] for row in rows)),
        "n_points": n_points,
    }
    return ({key: _as_json_value(value) for key, value in summary.items()}, rows)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--subset-run", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hq-threshold", type=float, default=0.4)
    args = parser.parse_args(argv)

    geometry = _load_npz(args.geometry_run / "grasp_labels.npz")
    subset = _load_npz(args.subset_run / "labels.npz")
    ids_path = args.candidate_ids or (args.subset_run / "candidate_ids.npy")
    ids = np.load(ids_path)
    subset_scores = subset["scores"]
    summary, rows = summarize_stratified_subset(
        collision=geometry["collision"],
        candidate_ids=ids,
        scores=subset_scores,
        hq_threshold=args.hq_threshold,
    )
    if summary["population_size_sum"] != int((~geometry["collision"]).sum()):
        raise RuntimeError("population size sum does not match full geometry-valid count")
    if summary["sample_size_sum"] != int(ids.size):
        raise RuntimeError("sample size sum does not match candidate-id manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "stratified_statistics.json"
    csv_path = args.output_dir / "stratified_statistics.csv"
    report = dict(summary)
    report["points"] = rows
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(rows[0].keys()) if rows else ["point_id"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
