"""Canonical pairwise sanity checks for the frozen V4 formula."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_score_v4 import compose_v4_score, stability_from_center_distance


def _score(normal, support, center_distance):
    stability = stability_from_center_distance(center_distance)
    return {
        "normal": float(normal),
        "support": float(support),
        "stability": stability,
        "total": compose_v4_score(normal, support, stability),
    }


def run_pairwise_checks():
    """Evaluate four hand-specified ranking cases from the V4 design review."""
    cases = [
        (
            "A_center_large_support_vs_edge_small_support",
            _score(0.98, 0.90, 0.05),
            _score(0.85, 0.20, 0.80),
            "A>B",
            "left must rank above right",
        ),
        (
            "B_slight_offset_good_antipodal_vs_center_bad_normals",
            _score(0.45, 0.25, 0.00),
            _score(0.95, 0.80, 0.20),
            "B>A",
            "good bilateral geometry must overcome small offset",
        ),
        (
            "C_large_support_vs_tiny_support",
            _score(0.90, 0.80, 0.20),
            _score(0.95, 0.05, 0.00),
            "A>B",
            "bilateral support must dominate tiny support",
        ),
        (
            "D_local_stable_handle_vs_center_weak_support",
            _score(0.95, 0.90, 0.50),
            _score(0.95, 0.08, 0.00),
            "A>B",
            "local valid handle grasp may beat weak central grasp",
        ),
    ]
    return [
        {
            "case": name,
            "score_a": a["total"],
            "score_b": b["total"],
            "expected": expected,
            "passed": bool(a["total"] > b["total"] if expected == "A>B" else b["total"] > a["total"]),
            "margin": a["total"] - b["total"],
            "note": note,
            "a_normal": a["normal"],
            "a_support": a["support"],
            "a_stability": a["stability"],
            "b_normal": b["normal"],
            "b_support": b["support"],
            "b_stability": b["stability"],
        }
        for name, a, b, expected, note in cases
    ]


def write_checks(path, checks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0].keys()))
        writer.writeheader()
        writer.writerows(checks)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", required=True)
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    checks = run_pairwise_checks()
    write_checks(args.output_csv, checks)
    failed = [check for check in checks if not check["passed"]]
    print(f"V4 pairwise checks: {len(checks) - len(failed)}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
