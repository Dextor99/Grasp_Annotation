"""Aggregate GN and Ours geometry-only results on the same source PLYs.

This report intentionally contains no force-closure or SDF fields.  It is the
common, reproducible geometry protocol used while watertight reconstruction is
not available for every object.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DETAIL_FIELDS = (
    "object", "method", "evaluation_mode", "input_source",
    "n_grasp_points", "n_raw_candidates", "n_unique_outputs",
    "n_geometry_valid", "geometry_valid_rate",
    "common_geometry_valid_rate_output", "common_geometry_yield_raw",
    "geometry_runtime_s", "native_generation_runtime_s", "common_geometry_eval_time_s",
)
SUMMARY_METRICS = (
    "n_raw_candidates", "n_geometry_valid", "geometry_valid_rate",
    "common_geometry_valid_rate_output", "common_geometry_yield_raw",
    "geometry_runtime_s", "native_generation_runtime_s", "common_geometry_eval_time_s",
)


def _load_summary(directory: Path) -> dict[str, Any]:
    path = Path(directory) / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing geometry summary: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return value


def _number(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int | None = None) -> int | None:
    number = _number(value)
    return default if number is None else int(number)


def _gn_row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    raw = _int(summary.get("n_candidates", summary.get("raw_candidate_count")), 0) or 0
    valid = _int(summary.get("n_geometry_valid"), 0) or 0
    rate = _number(summary.get("geometry_valid_rate"))
    if rate is None:
        rate = valid / raw if raw else 0.0
    timing = summary.get("timing") if isinstance(summary.get("timing"), dict) else {}
    runtime = _number(summary.get("geometry_runtime_s"))
    if runtime is None:
        runtime = sum(float(timing.get(key, 0.0) or 0.0) for key in ("preprocess", "candidate_generation", "width_collision"))
    return {
        "object": name, "method": "GN-geometry",
        "evaluation_mode": "ply_geometry_only", "input_source": summary.get("input_source", "surface_ply"),
        "n_grasp_points": _int(summary.get("n_grasp_points"), 0),
        "n_raw_candidates": raw, "n_unique_outputs": "",
        "n_geometry_valid": valid, "geometry_valid_rate": rate,
        "common_geometry_valid_rate_output": "", "common_geometry_yield_raw": "",
        "geometry_runtime_s": runtime, "native_generation_runtime_s": "", "common_geometry_eval_time_s": "",
    }


def _ours_row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    raw = _int(summary.get("n_raw_candidates", summary.get("n_candidates")), 0) or 0
    unique = _int(summary.get("n_unique_outputs"), 0) or 0
    valid = _int(summary.get("n_common_geometry_valid", summary.get("common_geometry_valid")), 0) or 0
    rate = _number(summary.get("common_geometry_valid_rate_output"))
    yield_raw = _number(summary.get("common_geometry_yield_raw"))
    return {
        "object": name, "method": "Ours-common-geometry",
        "evaluation_mode": summary.get("evaluation_mode", "ply_common_geometry"), "input_source": "surface_ply",
        "n_grasp_points": "", "n_raw_candidates": raw, "n_unique_outputs": unique,
        "n_geometry_valid": valid, "geometry_valid_rate": rate if rate is not None else (valid / unique if unique else 0.0),
        "common_geometry_valid_rate_output": rate if rate is not None else (valid / unique if unique else 0.0),
        "common_geometry_yield_raw": yield_raw if yield_raw is not None else (valid / raw if raw else 0.0),
        "geometry_runtime_s": "", "native_generation_runtime_s": _number(summary.get("native_generation_runtime_s"), 0.0),
        "common_geometry_eval_time_s": _number(summary.get("common_geometry_eval_time_s"), 0.0),
    }


def aggregate_ply_geometry(objects: dict[str, dict[str, Path]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for name, paths in objects.items():
        rows.append(_gn_row(name, _load_summary(paths["gn"])))
        rows.append(_ours_row(name, _load_summary(paths["ours"])))
    summary: dict[str, dict[str, Any]] = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        stats: dict[str, Any] = {"method": method, "n_objects": len(selected)}
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in selected if row[metric] not in (None, "")]
            stats[f"mean_{metric}"] = sum(values) / len(values) if values else ""
            if values:
                mean = stats[f"mean_{metric}"]
                stats[f"std_{metric}"] = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
            else:
                stats[f"std_{metric}"] = ""
        summary[method] = stats
    return rows, summary


def _parse_object_dirs(values: list[str]) -> dict[str, dict[str, Path]]:
    parsed: dict[str, dict[str, Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--object-dir must use NAME=GN_DIR,OURS_DIR")
        name, pair = value.split("=", 1)
        paths = pair.split(",", 1)
        if not name or len(paths) != 2 or not all(paths):
            raise ValueError(f"invalid --object-dir value: {value!r}")
        parsed[name] = {"gn": Path(paths[0]), "ours": Path(paths[1])}
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-dir", action="append", required=True, help="NAME=GN_SUMMARY_DIR,OURS_SUMMARY_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        objects = _parse_object_dirs(args.object_dir)
        rows, summary = aggregate_ply_geometry(objects)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = args.output_dir / "ply_geometry_all_objects.csv"
    with detail.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DETAIL_FIELDS))
        writer.writeheader(); writer.writerows(rows)
    summary_path = args.output_dir / "ply_geometry_summary.csv"
    fields = ["method", "n_objects"] + [f"{prefix}_{metric}" for metric in SUMMARY_METRICS for prefix in ("mean", "std")]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in sorted(summary):
            writer.writerow(summary[method])
    print(json.dumps({"objects": len(objects), "rows": len(rows), "all_objects": str(detail), "summary": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
