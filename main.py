"""Command-line entry point for automatic 6-DoF grasp annotation."""

from __future__ import annotations

import argparse

from grasp_config import GraspGenerationConfig
from grasp_export import export_grasp_annotations
from grasp_pipeline import run_grasp_annotation


def build_parser():
    parser = argparse.ArgumentParser(description="Generate finalized 6-DoF grasp annotations")
    parser.add_argument("--object", required=True, help="Input PLY point cloud")
    parser.add_argument("--views", type=int, default=5, help="Number of Fibonacci views")
    parser.add_argument("--anchors", type=int, default=3, help="Surface anchors per view")
    parser.add_argument("--mode", choices=("global", "normal", "cone"), default="cone")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--cone-angle", type=float, default=15.0)
    parser.add_argument("--approach-azimuth", type=int, default=4)
    parser.add_argument("--normal-knn", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    config = GraspGenerationConfig(
        num_views=args.views,
        anchors_per_view=args.anchors,
        mode=args.mode,
        cone_angle_deg=args.cone_angle,
        num_approach_azimuth=args.approach_azimuth,
        normal_knn=args.normal_knn,
        deterministic=not args.non_deterministic,
        random_seed=args.seed,
        enable_visualization=args.visualize,
    )
    result = run_grasp_annotation(args.object, config=config)
    paths = export_grasp_annotations(result, args.output)
    print(
        f"Finalized grasps: raw={result.meta['raw_grasp_count']}, "
        f"unique={result.meta['unique_grasp_count']}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
