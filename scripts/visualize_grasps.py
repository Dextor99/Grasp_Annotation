"""Visualize exported finalized 6-DoF grasp annotations with Open3D."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_visualization import (
    build_visualization_geometries,
    load_grasp_records,
    load_meta,
    print_grasp_details,
    print_visualization_summary,
    select_grasps,
    show_geometries,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, help="Input PLY used for grasp generation")
    parser.add_argument("--results", required=True, help="Directory containing grasps.json and meta.json")
    parser.add_argument("--topk", type=int, default=20, help="Number of highest-scoring grasps to display")
    parser.add_argument("--index", type=int, default=0, help="Ranked index for --mode single")
    parser.add_argument(
        "--mode",
        choices=("single", "overlay", "by_view", "by_anchor", "by_approach"),
        default="overlay",
    )
    parser.add_argument("--style", choices=("line", "mesh"), default="line")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--show-anchor", "--show-anchors", dest="show_anchor", action="store_true")
    parser.add_argument("--show-approach", action="store_true")
    parser.add_argument("--show-normal", action="store_true")
    parser.add_argument("--show-frame", action="store_true")
    parser.add_argument("--show-inner-points", action="store_true")
    parser.add_argument("--show-contacts", action="store_true")
    parser.add_argument(
        "--print-details",
        action="store_true",
        help="Print score, geometry, and source fields for displayed grasps",
    )
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--save-image", default=None, help="Optional screenshot path")
    return parser


def _print_selected_grasp(index, grasp):
    print("\n=== Selected grasp ===")
    print(f"Rank index: {index}")
    for field in ("score_total", "opening_mm", "depth_mm", "translation", "approach_direction"):
        print(f"{field}: {grasp[field]}")
    print(
        "source: "
        f"view={grasp['view_id']}, anchor={grasp['anchor_id']}, approach={grasp['approach_id']}"
    )


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    if args.topk < 0:
        raise ValueError("--topk must be >= 0")
    if args.point_size <= 0:
        raise ValueError("--point-size must be > 0")
    records = load_grasp_records(args.results)
    meta = load_meta(args.results)
    ranked = select_grasps(records, topk=None, score_threshold=args.score_threshold)
    if args.mode == "single":
        if not ranked:
            raise RuntimeError("No grasps available for visualization")
        if args.index < 0 or args.index >= len(ranked):
            raise IndexError(f"--index {args.index} out of range; {len(ranked)} grasps available")
        shown = [ranked[args.index]]
        color_by = "score"
        _print_selected_grasp(args.index, shown[0])
    else:
        shown = ranked[: args.topk]
        color_by = {"by_view": "view", "by_anchor": "anchor", "by_approach": "approach"}.get(
            args.mode, "score"
        )
    print_visualization_summary(records, shown, meta)
    if args.print_details:
        print_grasp_details(shown, start_rank=args.index if args.mode == "single" else 0)
    geometries, _ = build_visualization_geometries(
        args.object,
        shown,
        mode=args.mode,
        style=args.style,
        color_by=color_by,
        show_anchor=args.show_anchor,
        show_approach=args.show_approach,
        show_normal=args.show_normal,
        show_frame=args.show_frame,
        show_inner_points=args.show_inner_points,
        show_contacts=args.show_contacts,
    )
    show_geometries(
        geometries,
        title=f"Grasp Visualization - {Path(args.object).stem}",
        point_size=args.point_size,
        save_image=args.save_image,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
