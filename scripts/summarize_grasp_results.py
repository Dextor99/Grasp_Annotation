"""Summarize exported grasp result directories for experiment tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


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
    "closure_acceptance_rate",
    "merge_retention_rate",
    "total_s",
)


def _safe_rate(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def summarize_result_directory(result_directory):
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
    return {
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
        "total_s": (meta.get("timings") or {}).get("total_s"),
    }


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
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    rows = [summarize_result_directory(path) for path in args.results]
    _write_summary(rows, args.output, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
