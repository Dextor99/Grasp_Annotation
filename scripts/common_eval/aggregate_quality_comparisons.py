"""Build the fixed three-object force-closure quality comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "object", "method", "evaluation_mode", "sample_count", "n_raw_candidates",
    "n_unique_outputs", "fc_yield_raw", "hq_yield_raw", "hq_rate_among_fc",
    "mean_mu", "native_runtime_s",
)
METRICS = ("fc_yield_raw", "hq_yield_raw", "hq_rate_among_fc", "mean_mu")


def _load(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing quality summary: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"quality summary must be an object: {path}")
    return value


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        return default if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return default


def aggregate_quality(objects: dict[str, dict[str, Path]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for name, paths in objects.items():
        gn = _load(paths["gn"])
        rows.append({
            "object": name, "method": "GN-10k-weighted", "evaluation_mode": "weighted_stratified_10k",
            "sample_count": int(_num(gn.get("sample_count"), 0) or 0), "n_raw_candidates": int(_num(gn.get("raw_candidate_count"), 0) or 0),
            "n_unique_outputs": "", "fc_yield_raw": _num(gn.get("estimated_fc_yield_raw"), 0.0),
            "hq_yield_raw": _num(gn.get("estimated_hq_yield_raw"), 0.0), "hq_rate_among_fc": _num(gn.get("estimated_hq_rate_among_fc"), 0.0),
            "mean_mu": _num(gn.get("weighted_mean_mu"), -1.0), "native_runtime_s": "",
        })
        ours = _load(paths["ours"])
        rows.append({
            "object": name, "method": "Ours-all-unique", "evaluation_mode": "all_unique_outputs",
            "sample_count": "", "n_raw_candidates": int(_num(ours.get("n_raw_candidates", ours.get("n_candidates")), 0) or 0),
            "n_unique_outputs": int(_num(ours.get("n_unique_outputs"), 0) or 0), "fc_yield_raw": _num(ours.get("fc_yield_raw"), 0.0),
            "hq_yield_raw": _num(ours.get("hq_yield_raw"), 0.0), "hq_rate_among_fc": _num(ours.get("hq_rate_among_fc"), 0.0),
            "mean_mu": _num(ours.get("mean_mu"), -1.0), "native_runtime_s": _num(ours.get("native_wall_time_s"), 0.0),
        })
    summary: dict[str, dict[str, Any]] = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        out: dict[str, Any] = {"method": method, "n_objects": len(selected)}
        for metric in METRICS:
            vals = [float(row[metric]) for row in selected if row[metric] not in (None, "")]
            out[f"mean_{metric}"] = sum(vals) / len(vals) if vals else ""
            out[f"std_{metric}"] = (sum((v - out[f"mean_{metric}"]) ** 2 for v in vals) / len(vals)) ** 0.5 if vals else ""
        summary[method] = out
    return rows, summary


def _parse(values: list[str]) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--object-quality must use NAME=GN_STATS,OURS_SUMMARY")
        name, pair = value.split("=", 1); paths = pair.split(",", 1)
        if not name or len(paths) != 2 or not all(paths):
            raise ValueError(f"invalid --object-quality value: {value!r}")
        result[name] = {"gn": Path(paths[0]), "ours": Path(paths[1])}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-quality", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows, summary = aggregate_quality(_parse(args.object_quality))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = args.output_dir / "quality_comparison.csv"
    with detail.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    summary_path = args.output_dir / "quality_comparison_summary.csv"
    fields = ["method", "n_objects"] + [f"{prefix}_{metric}" for metric in METRICS for prefix in ("mean", "std")]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for method in sorted(summary): writer.writerow(summary[method])
    print(json.dumps({"rows": len(rows), "detail": str(detail), "summary": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
