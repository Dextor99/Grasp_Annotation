"""Re-score existing grasps with robust contact and centrality diagnostics.

This module is deliberately evaluation-only: it reads exported grasps and does
not regenerate candidates or change the frozen ``score_total`` field.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_visualization import load_grasp_records
from object_preprocess import prepare_object


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(length) or length < 1e-9:
        return np.zeros(3, dtype=float)
    return vector / length


def _mean_unit(vectors):
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) == 0:
        return np.zeros(3, dtype=float)
    vectors = np.asarray([_unit(vector) for vector in vectors])
    return _unit(np.sum(vectors, axis=0))


def _support_area(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points[:, [0, 2]]).volume)
    except Exception:
        return 0.0


def _normal_dispersion(normals, mean_normal):
    if len(normals) == 0 or np.linalg.norm(mean_normal) < 1e-9:
        return 1.0
    normalized = np.asarray([_unit(normal) for normal in normals])
    return float(np.mean(1.0 - np.clip(normalized @ mean_normal, -1.0, 1.0)))


def compute_diagnostic_metrics(record, object_data, finger_thickness_mm=5.0, finger_length_mm=100.0):
    """Compute robust bilateral, support and centrality metrics for one grasp."""
    points_world = np.asarray(object_data.points, dtype=float)
    normals_world = np.asarray(object_data.normals, dtype=float)
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.asarray(record["rotation_matrix"], dtype=float)
    pose[:3, 3] = np.asarray(record["translation"], dtype=float)
    object_world = np.asarray(object_data.T_object_world, dtype=float)
    gripper_world = object_world @ pose
    rotation = gripper_world[:3, :3]
    origin = gripper_world[:3, 3]
    points_local = (rotation.T @ (points_world - origin).T).T
    normals_local = (rotation.T @ normals_world.T).T

    opening = float(record.get("opening_mm", record.get("grasp_width_mm", 0.0)))
    opening = max(opening, 0.0)
    inner_mask = (
        (np.abs(points_local[:, 0]) <= finger_thickness_mm / 2.0)
        & (np.abs(points_local[:, 1]) <= opening / 2.0)
        & (points_local[:, 2] >= -finger_length_mm)
        & (points_local[:, 2] <= 0.0)
    )
    inner_points = points_local[inner_mask]
    inner_normals = normals_local[inner_mask]
    if len(inner_points) < 2:
        return {
            "contact_points_left": 0,
            "contact_points_right": 0,
            "contact_area_left_mm2": 0.0,
            "contact_area_right_mm2": 0.0,
            "normal_alignment_left": 0.0,
            "normal_alignment_right": 0.0,
            "normal_alignment_robust": 0.0,
            "normal_dispersion_left": 1.0,
            "normal_dispersion_right": 1.0,
            "support_score": 0.0,
            "center_distance_mm": float(np.linalg.norm(origin - np.asarray(object_data.center))),
            "center_distance_normalized": 0.0,
            "stability_score": 0.0,
            "diagnostic_score": 0.0,
        }

    y_values = inner_points[:, 1]
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    span = max(y_max - y_min, 1e-9)
    contact_band = max(2.5, 0.08 * max(opening, span))
    left_mask = y_values <= y_min + contact_band
    right_mask = y_values >= y_max - contact_band
    left_points, right_points = inner_points[left_mask], inner_points[right_mask]
    left_normals, right_normals = inner_normals[left_mask], inner_normals[right_mask]
    left_mean = _mean_unit(left_normals)
    right_mean = _mean_unit(right_normals)
    left_alignment = float(left_mean @ np.array([0.0, -1.0, 0.0]))
    right_alignment = float(right_mean @ np.array([0.0, 1.0, 0.0]))
    robust_alignment = min(left_alignment, right_alignment)
    normal_score = float(np.clip((robust_alignment + 1.0) / 2.0, 0.0, 1.0))

    reference_area = 15.0 * finger_length_mm
    left_area = _support_area(left_points)
    right_area = _support_area(right_points)
    support_score = math.sqrt(
        float(np.clip(left_area / reference_area, 0.0, 1.0))
        * float(np.clip(right_area / reference_area, 0.0, 1.0))
    )
    center_distance = float(np.linalg.norm(origin - np.asarray(object_data.center)))
    radius = float(getattr(object_data, "radius", 0.0))
    center_distance_normalized = center_distance / radius if radius > 1e-9 else 0.0
    stability_score = float(np.exp(-((center_distance_normalized / 0.6) ** 2)))
    diagnostic_score = normal_score * (0.5 + 0.3 * support_score + 0.2 * stability_score)
    return {
        "contact_points_left": int(np.count_nonzero(left_mask)),
        "contact_points_right": int(np.count_nonzero(right_mask)),
        "contact_area_left_mm2": left_area,
        "contact_area_right_mm2": right_area,
        "normal_alignment_left": left_alignment,
        "normal_alignment_right": right_alignment,
        "normal_alignment_robust": robust_alignment,
        "normal_dispersion_left": _normal_dispersion(left_normals, left_mean),
        "normal_dispersion_right": _normal_dispersion(right_normals, right_mean),
        "support_score": support_score,
        "center_distance_mm": center_distance,
        "center_distance_normalized": center_distance_normalized,
        "stability_score": stability_score,
        "diagnostic_score": diagnostic_score,
    }


def diagnose_records(records, object_data):
    """Return records augmented with diagnostic metrics and both rankings."""
    rows = []
    for index, record in enumerate(records):
        row = {
            "record_index": index,
            "old_score_total": float(record.get("score_total", 0.0)),
        }
        row.update(compute_diagnostic_metrics(record, object_data))
        rows.append(row)
    old_order = sorted(range(len(rows)), key=lambda index: rows[index]["old_score_total"], reverse=True)
    diagnostic_order = sorted(range(len(rows)), key=lambda index: rows[index]["diagnostic_score"], reverse=True)
    for rank, index in enumerate(old_order, start=1):
        rows[index]["old_rank"] = rank
    for rank, index in enumerate(diagnostic_order, start=1):
        rows[index]["diagnostic_rank"] = rank
    return rows


def summarize_diagnostics(rows, topk=20):
    """Compare old and diagnostic ranking on center/support metrics."""
    if not rows:
        raise ValueError("diagnostic rows cannot be empty")
    old_top = sorted(rows, key=lambda row: row["old_rank"])[:topk]
    diagnostic_top = sorted(rows, key=lambda row: row["diagnostic_rank"])[:topk]
    def average(items, key):
        return float(np.mean([row[key] for row in items])) if items else 0.0
    return {
        "grasp_count": len(rows),
        "old_top1_score": old_top[0]["old_score_total"],
        "diagnostic_top1_score": diagnostic_top[0]["diagnostic_score"],
        "old_topk_mean_score": average(old_top, "old_score_total"),
        "diagnostic_topk_mean_score": average(diagnostic_top, "diagnostic_score"),
        "old_topk_center_distance_normalized": average(old_top, "center_distance_normalized"),
        "diagnostic_topk_center_distance_normalized": average(diagnostic_top, "center_distance_normalized"),
        "old_topk_support_score": average(old_top, "support_score"),
        "diagnostic_topk_support_score": average(diagnostic_top, "support_score"),
        "old_topk_normal_alignment": average(old_top, "normal_alignment_robust"),
        "diagnostic_topk_normal_alignment": average(diagnostic_top, "normal_alignment_robust"),
    }


def _resolve_path(value, input_csv):
    path = Path(value)
    if path.exists():
        return path
    candidate = input_csv.parent / path
    return candidate if candidate.exists() else path


def analyze_csv(input_csv, output_dir, summary_csv=None, topk=20):
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    summaries = []
    object_cache = {}
    for source in source_rows:
        result_dir = _resolve_path(source["result_dir"], input_csv)
        object_path = _resolve_path(source["object"], input_csv)
        cache_key = str(object_path)
        if cache_key not in object_cache:
            object_cache[cache_key] = prepare_object(str(object_path))
        rows = diagnose_records(load_grasp_records(result_dir), object_cache[cache_key])
        name = Path(source.get("object", result_dir.name)).stem
        mode = ""
        meta_path = result_dir / "meta.json"
        if meta_path.is_file():
            mode = json.loads(meta_path.read_text(encoding="utf-8")).get("config", {}).get("mode", "")
        row_path = output_dir / f"{result_dir.name}_v4_diagnostics.csv"
        with row_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"result_dir": source["result_dir"], "object": source.get("object", ""), "mode": mode}
        summary.update(summarize_diagnostics(rows, topk=topk))
        summaries.append(summary)
    if summary_csv is not None and summaries:
        summary_csv = Path(summary_csv)
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--topk", type=int, default=20)
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    summaries = analyze_csv(args.input_csv, args.output_dir, args.summary_csv, args.topk)
    print(f"Diagnosed {len(summaries)} result directories -> {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
