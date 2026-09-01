"""Audit that V3 remains auxiliary while V4 controls final selection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _source(name):
    return (PROJECT_ROOT / name).read_text(encoding="utf-8")


def run_audit():
    pipeline = _source("grasp_pipeline.py")
    scoring = _source("grasp_scoring.py")
    refinement = _source("grasp_refinement.py")
    merge = _source("grasp_merge.py")
    schema = _source("grasp_schema.py")

    return [
        {
            "check": "v3_geometry_input_for_refinement",
            "passed": "score_grasp_candidates" in pipeline
            and "compute_grasp_scores_simple" in scoring
            and "inner_points_local" in refinement
            and pipeline.rfind("score_grasp_candidates") < pipeline.rfind("refine_grasp_closures"),
            "evidence": "V3 computes inner_points_local before closure refinement; this supplies geometry, not final ranking.",
        },
        {
            "check": "v3_not_used_for_merge_selection",
            "passed": "score_total_v3" not in merge and 'score_key="score_total_v4"' in pipeline,
            "evidence": "merge_grasp_candidates receives score_key=score_total_v4 and contains no V3 score key.",
        },
        {
            "check": "v3_not_used_for_threshold_filter",
            "passed": "score_total_v3" not in pipeline and "score_total_v3" not in merge,
            "evidence": "No V3 threshold, Top-K truncation, or candidate rejection appears in pipeline/merge.",
        },
        {
            "check": "v4_score_used_for_merge",
            "passed": "score_grasps_v4" in pipeline
            and pipeline.rfind("score_grasps_v4") < pipeline.rfind("merge_grasp_candidates")
            and 'score_key="score_total_v4"' in pipeline,
            "evidence": "V4 scoring occurs after post-refinement validation and before SE(3) merge.",
        },
        {
            "check": "v4_score_used_for_final_export",
            "passed": 'grasp.get("score_total_v4", grasp.get("score_total"))' in schema,
            "evidence": "normalize_grasp_record exports score_total from score_total_v4 when present.",
        },
        {
            "check": "v3_fields_preserved",
            "passed": '"score_total_v3"' in schema and '"score_total_v4"' in schema,
            "evidence": "Final records retain both score_total_v3 and score_total_v4.",
        },
    ]


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "evidence"])
        writer.writeheader()
        writer.writerows(report)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", required=True)
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    report = run_audit()
    write_report(args.output_csv, report)
    failed = [item for item in report if not item["passed"]]
    print(f"V3 dependency audit: {len(report) - len(failed)}/{len(report)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
