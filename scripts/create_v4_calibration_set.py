"""Create a human-label calibration template for the V4 HQ threshold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
    parser.add_argument("--per-group", type=int, default=15)
    parser.add_argument("--output-csv", required=True)
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    rows = build_calibration_rows([_parse_model(value) for value in args.model], args.per_group)
    write_rows(args.output_csv, rows)
    print(f"Created {len(rows)} unlabeled V4 calibration rows -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
