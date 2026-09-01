import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


class CommonComparisonTests(unittest.TestCase):
    def test_score_summary_uses_fc_valid_denominator_and_explicit_hq_yield(self):
        from scripts.common_eval.build_comparison import summarize_scores

        result = summarize_scores(
            method="GN-Full",
            n_candidates=100,
            n_geometry_valid=20,
            scores=np.array([-1.0, 0.2, 0.4, 0.8], dtype=float),
            common_eval_count=4,
            native_wall_time_s=3.0,
            common_eval_wall_time_s=1.0,
        )
        self.assertEqual(result["common_fc_valid"], 3)
        self.assertEqual(result["n_mu_le_04"], 2)
        self.assertAlmostEqual(result["hq_rate_mu04"], 2 / 3)
        self.assertAlmostEqual(result["hq_yield"], 2 / 100)
        self.assertEqual(result["n_raw_candidates"], 100)
        self.assertEqual(result["n_unique_outputs"], "")
        self.assertAlmostEqual(result["fc_yield_raw"], 3 / 100)
        self.assertAlmostEqual(result["hq_yield_raw"], 2 / 100)
        self.assertAlmostEqual(result["hq_rate_among_fc"], 2 / 3)

    def test_estimate_from_exact_geometry_rate_is_marked(self):
        from scripts.common_eval.build_comparison import estimate_full_from_subset

        result = estimate_full_from_subset(
            n_candidates=1000,
            n_geometry_valid=200,
            subset_fc_valid=50,
            subset_hq=10,
            subset_common_eval=100,
        )
        self.assertAlmostEqual(result["estimated_common_fc_valid"], 100.0)
        self.assertAlmostEqual(result["estimated_n_mu_le_04"], 20.0)
        self.assertAlmostEqual(result["estimated_hq_yield"], 0.02)
        self.assertTrue(result["is_estimate"])

    def test_weighted_statistics_replace_legacy_full_estimate(self):
        from scripts.common_eval.build_comparison import apply_stratified_statistics

        with TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            path.write_text(json.dumps({
                "weighted_fc_rate": 0.12,
                "weighted_hq_probability": 0.03,
                "weighted_hq_rate_among_fc": 0.25,
                "weighted_mean_mu": 0.51,
                "estimated_fc_valid": 240.0,
                "estimated_n_mu_le_04": 60.0,
                "estimated_fc_yield_raw": 0.024,
                "estimated_hq_yield_raw": 0.006,
                "estimated_hq_rate_among_fc": 0.25,
            }), encoding="utf-8")
            row = {"is_estimate": False}
            apply_stratified_statistics(row, path)
            self.assertTrue(row["is_estimate"])
            self.assertEqual(row["estimated_common_fc_valid"], 240.0)
            self.assertEqual(row["estimated_n_mu_le_04"], 60.0)
            self.assertEqual(row["weighted_mean_mu"], 0.51)


if __name__ == "__main__":
    unittest.main()
