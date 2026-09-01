"""Create a human-label calibration template for the V4 HQ threshold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load_records(result_dir):
    path = Path(result_dir) / "grasps.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list")
    return records


def build_calibration_rows(models, per_group=15):
    """Select top/bottom V4 candidates; leave human labels intentionally blank."""
    if not isinstance(per_group, int) or per_group <= 0:
        raise ValueError("per_group must be a positive integer")
    rows = []
    for object_name, result_dir in models:
        records = _load_records(result_dir)
        if len(records) < 2 * per_group:
            raise ValueError(f"{object_name} has fewer than {2 * per_group} grasps")
        ranked = sorted(
            enumerate(records),
            key=lambda item: float(item[1].get("score_total_v4", item[1].get("score_total", 0.0))),
            reverse=True,
        )
        selected = [(index, record, "top_candidate") for index, record in ranked[:per_group]]
        selected += [(index, record, "bottom_candidate") for index, record in ranked[-per_group:]]
        for index, record, group in selected:
            rows.append(
                {
                    "object": object_name,
                    "result_dir": str(result_dir),
                    "record_index": index,
                    "selection_group": group,
                    "score_total_v4": float(record.get("score_total_v4", record.get("score_total", 0.0))),
                    "human_label": "",
                    "notes": "",
                }
            )
    return rows


def build_stratified_calibration_rows(models, mode="quick", seed=0):
    """Build deterministic high/middle/low samples for quick or full review."""
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be quick or full")
    total, counts = (20, (7, 6, 7)) if mode == "quick" else (30, (10, 10, 10))
    rng = np.random.default_rng(int(seed))
    rows = []
    for object_name, result_dir in models:
        records = _load_records(result_dir)
        if len(records) < total:
            raise ValueError(f"{object_name} has fewer than {total} grasps")
        ranked = sorted(
            enumerate(records),
            key=lambda item: float(item[1].get("score_total_v4", item[1].get("score_total", 0.0))),
            reverse=True,
        )
        n = len(ranked)
        bins = (
            ranked[: max(1, int(np.ceil(0.25 * n)))],
            ranked[max(0, int(np.floor(0.25 * n))): int(np.ceil(0.75 * n))],
            ranked[int(np.floor(0.75 * n)):],
        )
        labels = ("high_candidate", "middle_candidate", "low_candidate")
        selected = []
        for bucket, count, label in zip(bins, counts, labels):
            if len(bucket) < count:
                raise ValueError(f"{object_name} does not have enough samples in {label} bin")
            indices = rng.choice(len(bucket), size=count, replace=False)
            selected.extend((bucket[int(index)][0], bucket[int(index)][1], label) for index in indices)
        # Keep the sheet order stable by stratum, then by V4 score descending.
        selected.sort(key=lambda item: (labels.index(item[2]), -float(item[1].get("score_total_v4", item[1].get("score_total", 0.0)))))
        for index, record, group in selected:
            rows.append(
                {
                    "object": object_name,
                    "result_dir": str(result_dir),
                    "record_index": index,
                    "selection_group": group,
                    "score_total_v4": float(record.get("score_total_v4", record.get("score_total", 0.0))),
                    "human_label": "",
                    "notes": "",
                }
            )
    for index, row in enumerate(rows, 1):
        row["sample_id"] = f"{index:03d}"
    return rows


def write_calibration_files(labels_path, manifest_path, rows):
    """Write a visible label sheet and a score-bearing manifest separately."""
    labels_path = Path(labels_path)
    manifest_path = Path(manifest_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    visible_fields = ["sample_id", "object", "record_index", "human_label", "notes"]
    hidden_fields = list(rows[0].keys())
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=visible_fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in visible_fields} for row in rows)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=hidden_fields)
        writer.writeheader()
        writer.writerows(rows)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_model(value):
    if "=" not in value:
        raise ValueError("--model must use name=result_directory")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("--model must use name=result_directory")
    return name, path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="name=result_directory; repeat for each object")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="20 samples/object: 7 high, 6 middle, 7 low")
    mode.add_argument("--full", action="store_true", help="30 samples/object: 10 high, 10 middle, 10 low")
    parser.add_argument("--per-group", type=int, default=None, help="legacy top/bottom mode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", required=True, help="visible labels CSV")
    parser.add_argument("--manifest-csv", default=None, help="hidden score manifest CSV")
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    models = [_parse_model(value) for value in args.model]
    if args.per_group is not None:
        rows = build_calibration_rows(models, args.per_group)
    else:
        rows = build_stratified_calibration_rows(models, mode="full" if args.full else "quick", seed=args.seed)
    manifest = args.manifest_csv or str(Path(args.output_csv).with_name("manifest.csv"))
    write_calibration_files(args.output_csv, manifest, rows)
    print(f"Created {len(rows)} unlabeled V4 calibration rows -> {args.output_csv}")
    print(f"Hidden score manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
