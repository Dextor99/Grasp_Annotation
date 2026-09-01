"""Build a paper-ready Ours/GN-Full common-evaluation comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


COMPARISON_FIELDS = (
    "method", "n_candidates", "n_geometry_valid", "geometry_valid_rate",
    "common_eval_count", "common_fc_valid", "common_fc_valid_rate",
    "n_mu_le_04", "hq_rate_mu04", "mean_mu", "hq_yield",
    "native_wall_time_s", "common_eval_wall_time_s",
    "is_estimate", "estimated_common_fc_valid", "estimated_n_mu_le_04",
    "estimated_hq_yield",
)


def summarize_scores(
    *, method: str, n_candidates: int, n_geometry_valid: int, scores: np.ndarray,
    common_eval_count: int, native_wall_time_s: float = 0.0,
    common_eval_wall_time_s: float = 0.0,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=float).reshape(-1)
    valid = np.isfinite(values) & (values >= 0.0)
    valid_scores = values[valid]
    hq = valid_scores <= 0.4
    common_eval_count = int(common_eval_count)
    return {
        "method": method,
        "n_candidates": int(n_candidates),
        "n_geometry_valid": int(n_geometry_valid),
        "geometry_valid_rate": float(n_geometry_valid / n_candidates) if n_candidates else 0.0,
        "common_eval_count": common_eval_count,
        "common_fc_valid": int(valid.sum()),
        "common_fc_valid_rate": float(valid.sum() / common_eval_count) if common_eval_count else 0.0,
        "n_mu_le_04": int(hq.sum()),
        "hq_rate_mu04": float(hq.mean()) if len(valid_scores) else 0.0,
        "mean_mu": float(valid_scores.mean()) if len(valid_scores) else -1.0,
        "hq_yield": float(hq.sum() / n_candidates) if n_candidates else 0.0,
        "native_wall_time_s": float(native_wall_time_s),
        "common_eval_wall_time_s": float(common_eval_wall_time_s),
        "is_estimate": False,
        "estimated_common_fc_valid": "",
        "estimated_n_mu_le_04": "",
        "estimated_hq_yield": "",
    }


def estimate_full_from_subset(
    *, n_candidates: int, n_geometry_valid: int, subset_fc_valid: int,
    subset_hq: int, subset_common_eval: int,
) -> dict[str, Any]:
    """Estimate full-run FC/HQ counts from exact geometry and a fixed subset.

    The returned values are explicitly marked estimates and must never be
    presented as exact full force-closure results.
    """
    subset_eval = max(1, int(subset_common_eval))
    fc_rate = float(subset_fc_valid) / subset_eval
    hq_rate = float(subset_hq) / max(1, int(subset_fc_valid))
    estimated_fc = float(n_geometry_valid) * fc_rate
    estimated_hq = estimated_fc * hq_rate
    return {
        "is_estimate": True,
        "estimated_common_fc_valid": estimated_fc,
        "estimated_n_mu_le_04": estimated_hq,
        "estimated_hq_yield": estimated_hq / int(n_candidates) if n_candidates else 0.0,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _elapsed_sum(directory: Path, pattern: str) -> float:
    total = 0.0
    for path in directory.glob(pattern):
        try:
            total += float(_load_json(path).get("elapsed_s", 0.0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return total


def _timing_total(directory: Path) -> float:
    path = directory / "timing.csv"
    if not path.is_file():
        return 0.0
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if str(row.get("stage", "")).lower() == "total":
                return float(row.get("seconds", 0.0))
    except (OSError, ValueError, TypeError):
        return 0.0
    return 0.0


def row_from_gn(geometry_run: Path, scored_run: Path, *, method: str, shard_dir: Path | None = None) -> dict[str, Any]:
    geometry_summary = _load_json(geometry_run / "summary.json")
    labels = np.load(scored_run / ("labels.npz" if (scored_run / "labels.npz").is_file() else "grasp_labels.npz"))
    scores = np.asarray(labels["scores"], dtype=float)
    geometry_count = int(geometry_summary.get("n_geometry_valid", geometry_summary.get("geometry_valid_count", 0)))
    n_candidates = int(geometry_summary.get("n_candidates", np.prod(scores.shape)))
    row = summarize_scores(
        method=method,
        n_candidates=n_candidates,
        n_geometry_valid=geometry_count,
        scores=scores,
        common_eval_count=geometry_count,
        native_wall_time_s=float(geometry_summary.get("timing", {}).get("total", _timing_total(geometry_run))),
        common_eval_wall_time_s=_elapsed_sum(shard_dir or scored_run, "fc_shard_*.json"),
    )
    return row


def row_from_ours_common(summary_path: Path) -> dict[str, Any]:
    summary = _load_json(summary_path)
    row = {field: summary.get(field, "") for field in COMPARISON_FIELDS}
    row["method"] = summary.get("method", "Ours-v1.2-common")
    return row


def write_comparison(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMPARISON_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--gn-np20-geometry", type=Path)
    parser.add_argument("--gn-np20-complete", type=Path)
    parser.add_argument("--gn-np20-shards", type=Path)
    parser.add_argument("--gn-full-geometry", type=Path)
    parser.add_argument("--gn-full-subset", type=Path)
    parser.add_argument("--gn-full-subset-shards", type=Path)
    parser.add_argument("--ours-common-summary", type=Path)
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    if args.gn_np20_geometry and args.gn_np20_complete:
        rows.append(row_from_gn(args.gn_np20_geometry, args.gn_np20_complete, method="GN-Full-Np20-exact", shard_dir=args.gn_np20_shards))
    if args.gn_full_geometry and args.gn_full_subset:
        geometry_summary = _load_json(args.gn_full_geometry / "summary.json")
        subset_summary = _load_json(args.gn_full_subset / "summary.json")
        labels = np.load(args.gn_full_subset / "labels.npz")
        base = summarize_scores(
            method="GN-Full-10k-subset",
            n_candidates=int(geometry_summary["n_candidates"]),
            n_geometry_valid=int(geometry_summary["n_geometry_valid"]),
            scores=np.asarray(labels["scores"]),
            common_eval_count=int(subset_summary["subset_count"]),
            native_wall_time_s=float(geometry_summary.get("timing", {}).get("total", _timing_total(args.gn_full_geometry))),
            common_eval_wall_time_s=_elapsed_sum(args.gn_full_subset_shards or args.gn_full_subset, "subset_fc_shard_*.json"),
        )
        base.update(estimate_full_from_subset(
            n_candidates=base["n_candidates"], n_geometry_valid=base["n_geometry_valid"],
            subset_fc_valid=base["common_fc_valid"], subset_hq=base["n_mu_le_04"],
            subset_common_eval=base["common_eval_count"],
        ))
        rows.append(base)
    if args.ours_common_summary:
        rows.append(row_from_ours_common(args.ours_common_summary))
    if not rows:
        parser.error("provide at least one complete method input")
    write_comparison(rows, args.output_csv)
    print(json.dumps({"rows": len(rows), "output": str(args.output_csv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
