import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.common_eval.aggregate_object_comparisons import aggregate_comparisons


def _write(path: Path, rows):
    fields = [
        "method", "n_raw_candidates", "n_unique_outputs", "n_geometry_valid",
        "geometry_valid_rate", "common_eval_count", "fc_yield_raw", "hq_yield_raw",
        "hq_rate_among_fc", "mean_mu", "native_wall_time_s", "common_eval_wall_time_s",
        "is_estimate", "estimated_fc_yield_raw", "estimated_hq_yield_raw",
        "estimated_hq_rate_among_fc", "weighted_mean_mu",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_uses_weighted_estimates_for_gn_and_direct_ours_metrics():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write(root / "cube.csv", [
            {"method": "GN-Full-10k-subset", "n_raw_candidates": 1000, "n_unique_outputs": "",
             "n_geometry_valid": 400, "geometry_valid_rate": .4, "common_eval_count": 10,
             "fc_yield_raw": .01, "hq_yield_raw": .002, "hq_rate_among_fc": .2,
             "mean_mu": .6, "native_wall_time_s": 10, "common_eval_wall_time_s": 2,
             "is_estimate": "True", "estimated_fc_yield_raw": .08,
             "estimated_hq_yield_raw": .02, "estimated_hq_rate_among_fc": .25,
             "weighted_mean_mu": .55},
            {"method": "Ours-v1.2-common", "n_raw_candidates": 100, "n_unique_outputs": 50,
             "n_geometry_valid": 30, "geometry_valid_rate": .6, "common_eval_count": 50,
             "fc_yield_raw": .1, "hq_yield_raw": .04, "hq_rate_among_fc": .4,
             "mean_mu": .3, "native_wall_time_s": 5, "common_eval_wall_time_s": 1,
             "is_estimate": "False", "estimated_fc_yield_raw": "",
             "estimated_hq_yield_raw": "", "estimated_hq_rate_among_fc": "",
             "weighted_mean_mu": ""},
        ])
        rows, summary = aggregate_comparisons({"cube": root / "cube.csv"})
        gn = next(row for row in rows if row["method"].startswith("GN"))
        ours = next(row for row in rows if row["method"].startswith("Ours"))
        assert gn["fc_yield_raw"] == .08
        assert gn["hq_yield_raw"] == .02
        assert gn["mean_mu"] == .55
        assert ours["fc_yield_raw"] == .1
        assert summary["GN-Full-10k-subset"]["n_objects"] == 1

