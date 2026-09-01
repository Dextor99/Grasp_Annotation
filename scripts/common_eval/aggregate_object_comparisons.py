"""Aggregate per-object Ours/GN common-evaluation CSV files.

This reporting-only utility never reruns geometry or force-closure evaluation.
For GN 10k rows it reports the weighted-stratified estimates; observed subset
values remain available in the source CSV and are not silently relabelled.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DETAIL_FIELDS = (
    "object", "method", "n_raw_candidates", "n_unique_outputs",
    "n_geometry_valid", "geometry_valid_rate", "common_eval_count",
    "fc_yield_raw", "hq_yield_raw", "hq_rate_among_fc", "mean_mu",
    "native_runtime_s", "common_eval_runtime_s", "evaluation_mode",
)
SUMMARY_METRICS = (
    "n_raw_candidates", "fc_yield_raw", "hq_yield_raw", "hq_rate_among_fc",
    "mean_mu", "native_runtime_s",
)


def _value(row: dict[str, str], key: str, default: Any = "") -> Any:
    raw = row.get(key, "")
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _normalized_row(object_name: str, source: dict[str, str]) -> dict[str, Any]:
    method = str(source.get("method", ""))
    is_gn = method.lower().startswith("gn")
    estimated = is_gn and _is_true(source.get("is_estimate", "")) and source.get("estimated_fc_yield_raw", "") != ""
    if estimated:
        fc_yield = _value(source, "estimated_fc_yield_raw", 0.0)
        hq_yield = _value(source, "estimated_hq_yield_raw", 0.0)
        hq_rate = _value(source, "estimated_hq_rate_among_fc", _value(source, "hq_rate_among_fc", 0.0))
        mean_mu = _value(source, "weighted_mean_mu", _value(source, "mean_mu", -1.0))
        mode = "weighted_stratified_10k"
    else:
        fc_yield = _value(source, "fc_yield_raw", 0.0)
        hq_yield = _value(source, "hq_yield_raw", 0.0)
        hq_rate = _value(source, "hq_rate_among_fc", 0.0)
        mean_mu = _value(source, "mean_mu", -1.0)
        mode = "full_exact" if is_gn and "exact" in method.lower() else ("common_official_all_unique" if not is_gn else "subset_observed")
    unique = source.get("n_unique_outputs", "")
    if unique != "":
        unique = int(float(unique))
    return {
        "object": object_name,
        "method": method,
        "n_raw_candidates": int(_value(source, "n_raw_candidates", _value(source, "n_candidates", 0))),
        "n_unique_outputs": unique,
        "n_geometry_valid": int(_value(source, "n_geometry_valid", 0)),
        "geometry_valid_rate": _value(source, "geometry_valid_rate", 0.0),
        "common_eval_count": int(_value(source, "common_eval_count", 0)),
        "fc_yield_raw": fc_yield,
        "hq_yield_raw": hq_yield,
        "hq_rate_among_fc": hq_rate,
        "mean_mu": mean_mu,
        "native_runtime_s": _value(source, "native_wall_time_s", 0.0),
        "common_eval_runtime_s": _value(source, "common_eval_wall_time_s", 0.0),
        "evaluation_mode": mode,
    }


def aggregate_comparisons(object_csvs: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for object_name, path in object_csvs.items():
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows.extend(_normalized_row(object_name, source) for source in csv.DictReader(handle))
    summary: dict[str, dict[str, Any]] = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        stats: dict[str, Any] = {"method": method, "n_objects": len(selected)}
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in selected]
            if not values:
                stats[f"mean_{metric}"] = ""
                stats[f"std_{metric}"] = ""
            else:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                stats[f"mean_{metric}"] = mean
                stats[f"std_{metric}"] = variance ** 0.5
        summary[method] = stats
    return rows, summary


def _parse_object_csv(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--object-csv must use OBJECT=CSV, got {value!r}")
        object_name, path = value.split("=", 1)
        if not object_name or not path:
            raise ValueError(f"invalid OBJECT=CSV value: {value!r}")
        parsed[object_name] = Path(path)
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-csv", action="append", required=True, help="OBJECT=per-object comparison CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = _parse_object_csv(args.object_csv)
    rows, summary = aggregate_comparisons(inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "comparison_all_objects.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DETAIL_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    summary_fields = ["method", "n_objects"] + [f"{prefix}_{metric}" for metric in SUMMARY_METRICS for prefix in ("mean", "std")]
    summary_path = args.output_dir / "comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for method in sorted(summary):
            writer.writerow(summary[method])
    print(json.dumps({"objects": len(inputs), "rows": len(rows), "all_objects": str(detail_path), "summary": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

