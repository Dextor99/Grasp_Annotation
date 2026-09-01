"""Compute score-distribution and high-quality grasp statistics from result CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_config import V4_HIGH_QUALITY_THRESHOLD

DEFAULT_THRESHOLDS = (0.0, V4_HIGH_QUALITY_THRESHOLD, 0.5, 0.8, 0.9, 0.95)


def _threshold_key(threshold):
    text = f"{float(threshold):g}"
    return text.replace("-", "m").replace(".", "_")


def compute_score_statistics(records, thresholds=DEFAULT_THRESHOLDS):
    """Return distribution, threshold-count and threshold-ratio statistics."""
    scores = np.asarray([float(record["score_total"]) for record in records], dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("at least one grasp with score_total is required")
    if not np.all(np.isfinite(scores)):
        raise ValueError("score_total values must be finite")

    stats = {
        "unique_grasp_count": int(len(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_std": float(np.std(scores)),
    }
    for percentile in (10, 25, 50, 75, 90, 95):
        stats[f"score_p{percentile}"] = float(np.percentile(scores, percentile))
    for threshold in thresholds:
        threshold = float(threshold)
        key = _threshold_key(threshold)
        count = int(np.count_nonzero(scores >= threshold))
        stats[f"score_ge_{key}_count"] = count
        stats[f"score_ge_{key}_ratio"] = float(count / len(scores))
    return stats


def _load_records(result_dir):
    path = Path(result_dir) / "grasps.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing grasp file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list")
    return records


def _resolve_result_dir(value, input_csv):
    path = Path(value)
    if path.is_dir():
        return path
    relative_to_csv = input_csv.parent / path
    if relative_to_csv.is_dir():
        return relative_to_csv
    return path


def _write_rows(output_csv, rows):
    output_csv = Path(output_csv)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_csv(input_csv, output_csv, aggregate_csv=None):
    """Analyze each result directory referenced by ``input_csv``."""
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    all_records = []
    total_generated = 0
    total_closure_valid = 0
    for source in source_rows:
        result_dir = _resolve_result_dir(source["result_dir"], input_csv)
        records = _load_records(result_dir)
        all_records.extend(records)
        total_generated += int(source.get("generated_candidate_count", 0) or 0)
        total_closure_valid += int(source.get("closure_valid_count", 0) or 0)
        row = {
            "result_dir": source["result_dir"],
            "object": source.get("object", ""),
            "generated_candidate_count": int(source.get("generated_candidate_count", 0) or 0),
            "closure_valid_count": int(source.get("closure_valid_count", 0) or 0),
            "high_quality_threshold": V4_HIGH_QUALITY_THRESHOLD,
        }
        meta_path = result_dir / "meta.json"
        if meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            config = meta.get("config", {})
            row.update(
                {
                    "mode": config.get("mode", ""),
                    "num_views": config.get("num_views", ""),
                    "anchors_per_view": config.get("anchors_per_view", ""),
                }
            )
        row.update(compute_score_statistics(records))
        row["high_quality_yield"] = float(
            row[f"score_ge_{_threshold_key(V4_HIGH_QUALITY_THRESHOLD)}_count"]
            / row["generated_candidate_count"]
            if row["generated_candidate_count"] else 0.0
        )
        rows.append(row)

    _write_rows(output_csv, rows)
    if aggregate_csv is not None and all_records:
        aggregate = {
            "result_dir": "__all__",
            "object": "ALL",
            "mode": "mixed",
            "num_views": "",
            "anchors_per_view": "",
            "generated_candidate_count": total_generated,
            "closure_valid_count": total_closure_valid,
            "high_quality_threshold": V4_HIGH_QUALITY_THRESHOLD,
        }
        aggregate.update(compute_score_statistics(all_records))
        aggregate["high_quality_yield"] = float(
            aggregate[f"score_ge_{_threshold_key(V4_HIGH_QUALITY_THRESHOLD)}_count"]
            / total_generated
            if total_generated else 0.0
        )
        _write_rows(aggregate_csv, [aggregate])
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="CSV containing a result_dir column")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--aggregate-csv", help="Optional weighted aggregate output CSV")
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    rows = analyze_csv(args.input_csv, args.output_csv, args.aggregate_csv)
    print(f"Analyzed {len(rows)} result directories -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
