import unittest


class V4ValidationTests(unittest.TestCase):
    def test_pairwise_canonical_cases_pass(self):
        from scripts.validate_v4_ranking import run_pairwise_checks

        checks = run_pairwise_checks()

        self.assertEqual(len(checks), 4)
        self.assertTrue(all(check["passed"] for check in checks))

    def test_component_summary_reports_topk_and_rank_correlations(self):
        from scripts.analyze_v4_components import summarize_component_rows

        rows = [
            {"v4_rank": "1", "score_total_v4": "0.9", "score_v4_normal": "0.9", "score_v4_support": "0.8", "score_v4_stability": "0.9"},
            {"v4_rank": "2", "score_total_v4": "0.6", "score_v4_normal": "0.7", "score_v4_support": "0.5", "score_v4_stability": "0.7"},
            {"v4_rank": "3", "score_total_v4": "0.3", "score_v4_normal": "0.4", "score_v4_support": "0.2", "score_v4_stability": "0.4"},
        ]

        summary, distribution = summarize_component_rows(rows, topks=(2, 3))

        self.assertEqual(summary["grasp_count"], 3)
        self.assertAlmostEqual(summary["top2_mean_score_v4"], 0.75)
        self.assertLess(summary["spearman_normal_vs_rank"], 0.0)
        self.assertEqual(distribution["score_total_v4_p50"], 0.6)


if __name__ == "__main__":
    unittest.main()
