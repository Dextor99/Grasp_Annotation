"""Summarize exported grasp result directories for experiment tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_config import V4_HIGH_QUALITY_THRESHOLD


SUMMARY_FIELDS = (
    "result_dir",
    "object",
    "generated_candidate_count",
    "scored_candidate_count",
    "refinement_input_count",
    "closure_geometry_rejected",
    "closure_pose_collision_rejected",
    "closure_valid_count",
    "unique_grasp_count",
    "high_quality_threshold",
    "high_quality_count",
    "high_quality_ratio",
    "high_quality_yield",
    "closure_acceptance_rate",
    "merge_retention_rate",
    "mean_score",
    "top1_score",
    "top20_mean_score",
    "generation_s",
    "scoring_s",
    "merge_s",
    "total_s",
)

HIGH_QUALITY_THRESHOLD = V4_HIGH_QUALITY_THRESHOLD


def _safe_rate(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _score_metrics(result_directory, high_quality_threshold=HIGH_QUALITY_THRESHOLD):
    path = Path(result_directory) / "grasps.json"
    if not path.is_file():
        return {
            "mean_score": None,
            "top1_score": None,
            "top20_mean_score": None,
            "high_quality_threshold": high_quality_threshold,
            "high_quality_count": 0,
            "high_quality_ratio": 0.0,
        }
    records = json.loads(path.read_text(encoding="utf-8"))
    scores = [
        float(record["score_total"])
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("score_total"), (int, float))
        and math.isfinite(float(record["score_total"]))
    ]
    if not scores:
        return {
            "mean_score": None,
            "top1_score": None,
            "top20_mean_score": None,
            "high_quality_threshold": high_quality_threshold,
            "high_quality_count": 0,
            "high_quality_ratio": 0.0,
        }
    top20 = scores[:20]
    return {
        "mean_score": sum(scores) / len(scores),
        "top1_score": scores[0],
        "top20_mean_score": sum(top20) / len(top20),
        "high_quality_threshold": high_quality_threshold,
        "high_quality_count": sum(score >= high_quality_threshold for score in scores),
        "high_quality_ratio": sum(score >= high_quality_threshold for score in scores) / len(scores),
    }


def summarize_result_directory(result_directory, high_quality_threshold=HIGH_QUALITY_THRESHOLD):
    """Read one ``meta.json`` and return stable experiment statistics."""
    result_directory = Path(result_directory)
    meta_path = result_directory / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    counts = meta.get("candidate_counts") or {}
    raw_count = int(meta.get("raw_grasp_count", 0))
    unique_count = int(meta.get("unique_grasp_count", 0))
    generated = int(counts.get("generated_candidate_count", raw_count))
    scored = int(counts.get("scored_candidate_count", generated))
    refinement_input = int(counts.get("refinement_input_count", scored))
    closure_valid = int(counts.get("closure_valid_count", raw_count))
    geometry_rejected = int(counts.get("closure_geometry_rejected", 0))
    pose_rejected = int(counts.get("closure_pose_collision_rejected", 0))
    timings = meta.get("timings") or {}
    summary = {
        "result_dir": str(result_directory),
        "object": meta.get("object"),
        "generated_candidate_count": generated,
        "scored_candidate_count": scored,
        "refinement_input_count": refinement_input,
        "closure_geometry_rejected": geometry_rejected,
        "closure_pose_collision_rejected": pose_rejected,
        "closure_valid_count": closure_valid,
        "unique_grasp_count": unique_count,
        "closure_acceptance_rate": _safe_rate(closure_valid, refinement_input),
        "merge_retention_rate": _safe_rate(unique_count, closure_valid),
        "generation_s": timings.get("generation_s"),
        "scoring_s": timings.get("scoring_s"),
        "merge_s": timings.get("merge_s"),
        "total_s": timings.get("total_s"),
    }
    summary.update(_score_metrics(result_directory, high_quality_threshold))
    summary["high_quality_yield"] = _safe_rate(
        summary["high_quality_count"], generated
    )
    return summary


def _write_summary(rows, output_path, output_format):
    if output_format == "json":
        payload = json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False)
        if output_path:
            Path(output_path).write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return
    if output_path:
        with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="Result directories containing meta.json")
    parser.add_argument("--output", default=None, help="Optional CSV/JSON output path")
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    parser.add_argument("--hq-threshold", type=float, default=HIGH_QUALITY_THRESHOLD)
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    if not math.isfinite(args.hq_threshold):
        raise ValueError("--hq-threshold must be finite")
    rows = [summarize_result_directory(path, args.hq_threshold) for path in args.results]
    _write_summary(rows, args.output, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
