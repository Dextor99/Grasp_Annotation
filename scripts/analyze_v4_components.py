"""Analyze V4 component contributions and score distributions offline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COMPONENTS = ("score_v4_normal", "score_v4_support", "score_v4_stability", "score_total_v4")


def _values(rows, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _rank_correlation(rows, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    ranks = np.asarray([float(row["v4_rank"]) for row in rows], dtype=float)
    if len(values) < 2 or np.all(values == values[0]):
        return 0.0
    result = spearmanr(values, ranks).statistic
    return float(result) if np.isfinite(result) else 0.0


def summarize_component_rows(rows, topks=(20, 100)):
    """Return component contribution summary and percentile distribution."""
    if not rows:
        raise ValueError("component rows cannot be empty")
    ordered = sorted(rows, key=lambda row: int(row["v4_rank"]))
    summary = {"grasp_count": len(ordered)}
    for topk in topks:
        selected = ordered[: int(topk)]
        prefix = f"top{int(topk)}"
        for field, suffix in (
            ("score_total_v4", "mean_score_v4"),
            ("score_v4_normal", "mean_normal"),
            ("score_v4_support", "mean_support"),
            ("score_v4_stability", "mean_stability"),
        ):
            summary[f"{prefix}_{suffix}"] = float(np.mean(_values(selected, field))) if selected else 0.0
    for field in COMPONENTS[:3]:
        summary[f"spearman_{field.removeprefix('score_v4_')}_vs_rank"] = _rank_correlation(ordered, field)

    distribution = {"grasp_count": len(ordered)}
    for field in COMPONENTS:
        values = _values(ordered, field)
        if len(values) == 0:
            values = np.zeros(1, dtype=float)
        percentiles = np.percentile(values, [10, 25, 50, 75, 90, 95])
        name = field
        distribution[f"{name}_min"] = float(np.min(values))
        distribution[f"{name}_p10"] = float(percentiles[0])
        distribution[f"{name}_p25"] = float(percentiles[1])
        distribution[f"{name}_p50"] = float(percentiles[2])
        distribution[f"{name}_p75"] = float(percentiles[3])
        distribution[f"{name}_p90"] = float(percentiles[4])
        distribution[f"{name}_p95"] = float(percentiles[5])
        distribution[f"{name}_max"] = float(np.max(values))
        distribution[f"{name}_std"] = float(np.std(values))
    return summary, distribution


def analyze_directory(input_dir, component_csv, distribution_csv, topks=(20, 100)):
    input_dir = Path(input_dir)
    component_rows, distribution_rows = [], []
    for path in sorted(input_dir.glob("*_v4_rescore.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        summary, distribution = summarize_component_rows(rows, topks=topks)
        object_name = path.name.removesuffix("_v4_rescore.csv")
        component_rows.append({"object": object_name, **summary})
        distribution_rows.append({"object": object_name, **distribution})
    if not component_rows:
        raise ValueError(f"no *_v4_rescore.csv files found in {input_dir}")
    for destination, values in ((component_csv, component_rows), (distribution_csv, distribution_rows)):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0].keys()))
            writer.writeheader()
            writer.writerows(values)
    return component_rows, distribution_rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--component-csv", required=True)
    parser.add_argument("--distribution-csv", required=True)
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    components, _ = analyze_directory(args.input_dir, args.component_csv, args.distribution_csv)
    print(f"Analyzed V4 components for {len(components)} objects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
