import json
import tempfile
import unittest
from pathlib import Path


class QualityComparisonAggregationTests(unittest.TestCase):
    def test_aggregate_uses_weighted_gn_and_all_ours_fields(self):
        from scripts.common_eval.aggregate_quality_comparisons import aggregate_quality

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); gn = root / "gn.json"; ours = root / "ours.json"
            gn.write_text(json.dumps({
                "estimated_fc_yield_raw": 0.1,
                "estimated_hq_yield_raw": 0.02,
                "estimated_hq_rate_among_fc": 0.2,
                "weighted_mean_mu": 0.5,
                "sample_count": 10000,
            }), encoding="utf-8")
            ours.write_text(json.dumps({
                "fc_yield_raw": 0.12, "hq_yield_raw": 0.03,
                "hq_rate_among_fc": 0.25, "mean_mu": 0.45,
                "n_raw_candidates": 100, "n_unique_outputs": 90,
                "native_wall_time_s": 2.0,
            }), encoding="utf-8")
            rows, summary = aggregate_quality({"cube": {"gn": gn, "ours": ours}})
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(next(r for r in rows if r["method"] == "GN-10k-weighted")["fc_yield_raw"], 0.1)
        self.assertAlmostEqual(next(r for r in rows if r["method"] == "Ours-all-unique")["hq_yield_raw"], 0.03)
        self.assertEqual(summary["GN-10k-weighted"]["n_objects"], 1)

    def test_aggregate_rejects_missing_inputs(self):
        from scripts.common_eval.aggregate_quality_comparisons import aggregate_quality
        with self.assertRaises(FileNotFoundError):
            aggregate_quality({"cube": {"gn": Path("missing"), "ours": Path("missing2")}})


if __name__ == "__main__":
    unittest.main()
