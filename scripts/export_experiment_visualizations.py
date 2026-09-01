"""Export headless Matplotlib line-frame visualizations for grasp experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_visualization import load_grasp_records, make_gripper_lineset, select_grasps
from object_preprocess import prepare_object


def wireframe_segments(record, object_data):
    """Return line-frame segments in the same world coordinates as the object."""
    line_set = make_gripper_lineset(record, object_data=object_data)
    points = np.asarray(line_set.points, dtype=float)
    lines = np.asarray(line_set.lines, dtype=np.int64)
    return points[lines]


def _set_equal_axes(axis, points):
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def export_grasp_plot(object_path, result_directory, output_path, topk, score_threshold=None, dpi=300):
    """Save one object point-cloud plot with the highest-scoring ``topk`` grasps."""
    object_data = prepare_object(str(object_path))
    records = select_grasps(
        load_grasp_records(result_directory),
        topk=topk,
        score_threshold=score_threshold,
    )
    object_points = np.asarray(object_data.points, dtype=float)
    if len(object_points) > 20000:
        step = int(np.ceil(len(object_points) / 20000))
        object_points = object_points[::step]

    figure = plt.figure(figsize=(8, 7), dpi=int(dpi))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        object_points[:, 0], object_points[:, 1], object_points[:, 2],
        s=1.0, c="lightgray", alpha=0.35, depthshade=False,
    )
    scores = np.asarray([float(record["score_total"]) for record in records], dtype=float)
    score_min = float(np.min(scores)) if len(scores) else 0.0
    score_span = float(np.ptp(scores)) if len(scores) else 1.0
    for index, record in enumerate(records):
        color_value = (scores[index] - score_min) / max(score_span, 1e-12)
        color = plt.cm.viridis(color_value)
        segments = wireframe_segments(record, object_data)
        for segment in segments:
            axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, linewidth=0.8)
    all_points = [object_points]
    for record in records:
        all_points.append(wireframe_segments(record, object_data).reshape(-1, 3))
    _set_equal_axes(axis, np.vstack(all_points))
    axis.set_title(f"{Path(object_path).stem}: Top-{len(records)} grasps")
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if args.score_threshold is not None and not np.isfinite(args.score_threshold):
        raise ValueError("--score-threshold must be finite")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    stem = Path(args.object).stem
    export_grasp_plot(
        args.object, args.results,
        Path(args.output_dir) / f"{stem}_top1.png", topk=1,
        score_threshold=args.score_threshold, dpi=args.dpi,
    )
    export_grasp_plot(
        args.object, args.results,
        Path(args.output_dir) / f"{stem}_top{args.topk}.png", topk=args.topk,
        score_threshold=args.score_threshold, dpi=args.dpi,
    )
    threshold_text = "all scores" if args.score_threshold is None else f"score >= {args.score_threshold:g}"
    print(f"Saved {stem} Top-1 and Top-{args.topk} visualizations ({threshold_text}) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
