"""Run the real-model freeze acceptance matrix for the grasp generator."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_config import GraspGenerationConfig
from grasp_export import export_grasp_annotations
from grasp_freeze_validation import (
    assert_annotation_invariants,
    assert_repeated_results_equal,
)
from grasp_pipeline import run_grasp_annotation


@dataclass(frozen=True)
class ValidationCase:
    label: str
    model: str
    views: int
    anchors: int


CASES = (
    ValidationCase("juxing_v3_a2", "juxing.ply", 3, 2),
    ValidationCase("yuanzhu_v3_a2", "yuanzhu.ply", 3, 2),
    ValidationCase("shuilongtou_v3_a2", "shuilongtou.ply", 3, 2),
    ValidationCase("shuilongtou_v5_a3", "shuilongtou.ply", 5, 3),
)


def run_validation_case(case, model_directory, output_directory, repeats=3, top_k=10):
    model_path = Path(model_directory) / case.model
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    config = GraspGenerationConfig(
        num_views=case.views,
        anchors_per_view=case.anchors,
        mode="cone",
    )
    case_start = perf_counter()
    reference = None
    repeat_summaries = []
    failures = []
    for repeat_id in range(repeats):
        result = run_grasp_annotation(model_path, config=config)
        try:
            assert_annotation_invariants(result)
        except Exception as error:
            failures.append(
                f"repeat {repeat_id} invariants: {type(error).__name__}: {error}"
            )
        if reference is None:
            reference = result
            export_grasp_annotations(reference, Path(output_directory) / case.label)
        else:
            try:
                assert_repeated_results_equal(reference, result, top_k=top_k)
            except Exception as error:
                failures.append(
                    f"repeat {repeat_id} reproducibility: {type(error).__name__}: {error}"
                )
        repeat_summaries.append(
            {
                "repeat_id": repeat_id,
                "raw_grasp_count": len(result.raw_grasps),
                "unique_grasp_count": len(result.unique_grasps),
                "candidate_counts": result.meta.get("candidate_counts", {
                    "generated_candidate_count": len(result.raw_grasps),
                    "scored_candidate_count": len(result.raw_grasps),
                    "refinement_input_count": len(result.raw_grasps),
                    "closure_geometry_rejected": 0,
                    "closure_pose_collision_rejected": 0,
                    "closure_valid_count": len(result.raw_grasps),
                    "unique_grasp_count": len(result.unique_grasps),
                }),
                "merge_reduction_ratio": result.meta.get("merge_reduction_ratio"),
                "total_s": result.meta["timings"]["total_s"],
            }
        )
    return {
        "label": case.label,
        "model": str(model_path),
        "views": case.views,
        "anchors": case.anchors,
        "approaches_per_anchor": config.num_approach_directions,
        "status": "failed" if failures else "passed",
        "wall_time_s": perf_counter() - case_start,
        "repeats": repeat_summaries,
        "failures": failures,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=str(PROJECT_ROOT / "model"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "freeze-validation"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.label for case in CASES],
        help="Run selected case(s); default runs the full acceptance matrix",
    )
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    if args.repeats < 2:
        raise ValueError("freeze validation requires at least two repeated runs")
    selected = [case for case in CASES if not args.case or case.label in args.case]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "passed", "cases": []}
    for case in selected:
        print(f"Running {case.label}: {case.model}, {case.views} views x {case.anchors} anchors x 5 approaches")
        try:
            case_summary = run_validation_case(
                case,
                args.models,
                output,
                repeats=args.repeats,
                top_k=args.top_k,
            )
            if case_summary["status"] != "passed":
                summary["status"] = "failed"
        except Exception as error:
            case_summary = {
                "label": case.label,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            summary["status"] = "failed"
        summary["cases"].append(case_summary)
        print(json.dumps(case_summary, ensure_ascii=False, indent=2, allow_nan=False))
    summary_path = output / "validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Validation summary: {summary_path}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
