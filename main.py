"""Command-line entry point for multi-view grasp annotation generation."""

import argparse

from grasp_database import save_grasp_dataset
from multi_view_grasp import generate_multi_view_grasps


def build_parser():
    parser = argparse.ArgumentParser(description="Generate a multi-view grasp dataset.")
    parser.add_argument("--object", required=True, dest="object_path")
    parser.add_argument("--views", type=int, choices=[20, 40, 60, 100], default=60)
    parser.add_argument("--output", required=True)
    parser.add_argument("--position-threshold-mm", type=float, default=5.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=10.0)
    return parser


def run(argv=None):
    args = build_parser().parse_args(argv)
    result = generate_multi_view_grasps(
        args.object_path, args.views, args.position_threshold_mm, args.rotation_threshold_deg
    )
    paths = save_grasp_dataset(result.grasps, args.output, {
        "object": args.object_path,
        "views": args.views,
        "position_threshold_mm": args.position_threshold_mm,
        "rotation_threshold_deg": args.rotation_threshold_deg,
    })
    raw_count = sum(result.view_candidate_counts.values())
    print(f"processed views: {len(result.view_candidate_counts)}")
    print(f"skipped views: {len(result.skipped_views)}")
    print(f"raw grasps: {raw_count}")
    print(f"deduplicated grasps: {len(result.grasps)}")
    print(f"saved grasps: {paths.saved_count}")
    print(f"per-view candidate counts: {result.view_candidate_counts}")
    print(f"output: {paths.json_path}, {paths.npz_path}, {paths.meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
