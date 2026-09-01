"""Offline re-ranking with the formal V4 scorer.

The script consumes existing ``grasps.json`` exports.  It never regenerates
grasps and never overwrites the legacy ``score_total``; the V4 ranking is an
explicit comparison artifact until the canonical validation is accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_score_v4 import score_grasp_v4
from grasp_visualization import load_grasp_records
from object_preprocess import prepare_object


def _resolve_path(value, input_csv):
    path = Path(value)
    candidates = [path, PROJECT_ROOT / path, input_csv.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _mean(rows, field):
    values = [float(row[field]) for row in rows if np.isfinite(float(row[field]))]
    return float(np.mean(values)) if values else 0.0


def rescore_records(records, object_data, topk=20):
    """Return V4-augmented rows and a ranking summary for one object."""
    if topk <= 0:
        raise ValueError("topk must be positive")
    scored = [score_grasp_v4(record, object_data) for record in records]
    old_order = sorted(
        range(len(scored)), key=lambda i: float(scored[i].get("score_total_v3", 0.0)), reverse=True
    )
    v4_order = sorted(
        range(len(scored)), key=lambda i: float(scored[i].get("score_total_v4", 0.0)), reverse=True
    )
    old_rank = {index: rank for rank, index in enumerate(old_order, 1)}
    v4_rank = {index: rank for rank, index in enumerate(v4_order, 1)}
    rows = []
    for index, grasp in enumerate(scored):
        row = {
            "record_index": index,
            "old_rank": old_rank[index],
            "v4_rank": v4_rank[index],
            "score_total_v3": float(grasp.get("score_total_v3", 0.0)),
        }
        for field in (
            "score_total_v4",
            "score_v4_normal",
            "score_v4_support",
            "score_v4_stability",
            "score_v4_normal_dispersion",
            "normal_alignment_left",
            "normal_alignment_right",
            "contact_points_left",
            "contact_points_right",
            "contact_area_left_mm2",
            "contact_area_right_mm2",
            "center_distance_normalized",
            "contact_band_mm",
        ):
            row[field] = grasp.get(field, 0.0)
        rows.append(row)
    old_top = sorted(rows, key=lambda row: row["old_rank"])[:topk]
    v4_top = sorted(rows, key=lambda row: row["v4_rank"])[:topk]
    old_top_ids = {row["record_index"] for row in old_top}
    v4_top_ids = {row["record_index"] for row in v4_top}
    summary = {
        "grasp_count": len(rows),
        "old_top1_record_index": old_top[0]["record_index"] if old_top else -1,
        "v4_top1_record_index": v4_top[0]["record_index"] if v4_top else -1,
        "old_topk_mean_score": _mean(old_top, "score_total_v3"),
        "v4_topk_mean_score": _mean(v4_top, "score_total_v4"),
        "old_topk_center_distance_normalized": _mean(old_top, "center_distance_normalized"),
        "v4_topk_center_distance_normalized": _mean(v4_top, "center_distance_normalized"),
        "old_topk_support": _mean(old_top, "score_v4_support"),
        "v4_topk_support": _mean(v4_top, "score_v4_support"),
        "old_topk_normal": _mean(old_top, "score_v4_normal"),
        "v4_topk_normal": _mean(v4_top, "score_v4_normal"),
        "topk_overlap": len(old_top_ids & v4_top_ids),
    }
    return rows, summary


def analyze_csv(input_csv, output_dir, summary_csv, topk=20):
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    summaries = []
    cache = {}
    for source in source_rows:
        result_dir = _resolve_path(source["result_dir"], input_csv)
        object_path = _resolve_path(source["object"], input_csv)
        cache_key = str(object_path.resolve())
        if cache_key not in cache:
            cache[cache_key] = prepare_object(str(object_path))
        records = load_grasp_records(result_dir)
        rows, summary = rescore_records(records, cache[cache_key], topk=topk)
        result_name = result_dir.name
        per_grasp_path = output_dir / f"{result_name}_v4_rescore.csv"
        with per_grasp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["record_index"])
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "result_dir": source.get("result_dir", ""),
            "object": source.get("object", ""),
            "mode": source.get("mode", ""),
            **summary,
        }
        summaries.append(summary)
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if summaries:
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
    summaries = analyze_csv(args.input_csv, args.output_dir, args.summary_csv, args.topk)
    print(f"V4-rescored {len(summaries)} result directories -> {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
