import unittest
from unittest.mock import patch

import numpy as np


class OursPlyCommonGeometryTests(unittest.TestCase):
    def test_evaluator_reports_geometry_metrics_without_force_closure(self):
        from scripts.common_eval.ours_ply_common_geometry import evaluate_ours_ply_geometry

        records = [
            {"translation": [0.0, 0.0, 0.0], "rotation_matrix": np.eye(3).tolist(), "depth_mm": 10.0, "opening_mm": 20.0,
             "score_total": 0.8},
            {"translation": [1.0, 0.0, 0.0], "rotation_matrix": np.eye(3).tolist(), "depth_mm": 10.0, "opening_mm": 20.0,
             "score_total": 0.7},
        ]
        with patch("scripts.common_eval.ours_ply_common_geometry.evaluate_official_collision",
                   return_value=(np.array([False, True]), np.array([False, False]))) as collision:
            summary, rows = evaluate_ours_ply_geometry(
                records, np.zeros((8, 3), dtype=np.float32), native_raw_count=10, native_unique_count=2,
            )
        collision.assert_called_once()
        self.assertEqual(summary["n_unique_outputs"], 2)
        self.assertEqual(summary["n_common_geometry_valid"], 1)
        self.assertAlmostEqual(summary["common_geometry_valid_rate_output"], 0.5)
        self.assertAlmostEqual(summary["common_geometry_yield_raw"], 0.1)
        self.assertEqual(summary["evaluation_mode"], "ply_common_geometry")
        self.assertEqual(rows[0]["geometry_valid"], True)
        self.assertEqual(rows[1]["geometry_valid"], False)

    def test_evaluator_rejects_nonfinite_points(self):
        from scripts.common_eval.ours_ply_common_geometry import evaluate_ours_ply_geometry

        with self.assertRaisesRegex(ValueError, "model_points_m"):
            evaluate_ours_ply_geometry([], np.array([[np.nan, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
