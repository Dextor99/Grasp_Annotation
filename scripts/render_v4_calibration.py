"""Render unlabeled V4 calibration samples and contact sheets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_visualization import load_grasp_records, make_gripper_lineset
from object_preprocess import prepare_object


def _resolve(value):
    path = Path(value)
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else path


def _set_equal_axes(axis, points):
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _draw_sample(axis, object_data, record, elev, azim):
    points = np.asarray(object_data.points, dtype=float)
    if len(points) > 3000:
        points = points[np.linspace(0, len(points) - 1, 3000, dtype=int)]
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.0, c="lightgray", alpha=0.42, depthshade=False)
    line_set = make_gripper_lineset(record, color=(0.9, 0.12, 0.08), object_data=object_data)
    line_points = np.asarray(line_set.points, dtype=float)
    for start, end in np.asarray(line_set.lines, dtype=int):
        segment = line_points[[start, end]]
        axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#d62728", linewidth=1.8)
    _set_equal_axes(axis, np.vstack([points, line_points]))
    axis.view_init(elev=elev, azim=azim)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])


def render_sample(row, object_data, record, output_path):
    """Render one sample with 3/4 and side views, without score annotations."""
    figure = plt.figure(figsize=(8, 4), dpi=160)
    left = figure.add_subplot(121, projection="3d")
    right = figure.add_subplot(122, projection="3d")
    _draw_sample(left, object_data, record, elev=25, azim=35)
    _draw_sample(right, object_data, record, elev=10, azim=90)
    left.set_title("3/4 view")
    right.set_title("side view")
    figure.suptitle(str(row["sample_id"]), fontsize=13, fontweight="bold")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_calibration(manifest_csv, output_dir, per_sheet=10):
    manifest_csv = Path(manifest_csv)
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    sheet_dir = output_dir / "sheets"
    image_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cache = {}
    records_cache = {}
    rendered = []
    for row in rows:
        result_dir = _resolve(row["result_dir"])
        key = str(result_dir.resolve())
        if key not in cache:
            meta_path = result_dir / "meta.json"
            if not meta_path.is_file():
                raise FileNotFoundError(meta_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cache[key] = prepare_object(str(_resolve(meta["object"])))
            records_cache[key] = load_grasp_records(result_dir)
        record = records_cache[key][int(row["record_index"])]
        rendered.append(render_sample(row, cache[key], record, image_dir / f"sample_{row['sample_id']}.png"))
    for offset in range(0, len(rendered), per_sheet):
        page = rendered[offset: offset + per_sheet]
        figure, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = np.asarray(axes).reshape(-1)
        for axis, image_path in zip(axes, page):
            axis.imshow(plt.imread(image_path))
            axis.set_title(image_path.stem.removeprefix("sample_"))
            axis.axis("off")
        for axis in axes[len(page):]:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(sheet_dir / f"calibration_sheet_{offset // per_sheet + 1:02d}.png", dpi=160)
        plt.close(figure)
    return rendered


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-sheet", type=int, default=10)
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    if args.per_sheet <= 0 or args.per_sheet > 10:
        raise ValueError("--per-sheet must be between 1 and 10")
    rendered = render_calibration(args.manifest_csv, args.output_dir, args.per_sheet)
    print(f"Rendered {len(rendered)} calibration samples -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
