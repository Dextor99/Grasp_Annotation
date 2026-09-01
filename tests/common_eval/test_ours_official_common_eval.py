import unittest

import numpy as np


class OursOfficialCommonEvalTests(unittest.TestCase):
    def test_record_conversion_maps_ours_axes_to_official_pose(self):
        from scripts.common_eval.ours_official_common_eval import record_to_official_components

        record = {
            "translation": [100.0, 200.0, 300.0],
            "rotation_matrix": np.eye(3).tolist(),
            "depth_mm": 20.0,
            "grasp_width_mm": 40.0,
        }
        converted = record_to_official_components(record)

        np.testing.assert_allclose(converted["grasp_point_m"], [0.1, 0.2, 0.28])
        np.testing.assert_allclose(
            converted["rotation"],
            np.asarray([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        np.testing.assert_allclose(converted["center_m"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(converted["grasp_row"][13:16], [0.1, 0.2, 0.28])
        self.assertEqual(converted["width_m"], 0.04)
        self.assertEqual(converted["depth_m"], 0.02)

    def test_record_conversion_applies_ours_object_frame_transform(self):
        from scripts.common_eval.ours_official_common_eval import record_to_official_components

        transform = np.eye(4)
        transform[:3, 3] = [100.0, 10.0, 25.0]
        record = {
            "translation": [0.0, 0.0, 0.0],
            "rotation_matrix": np.eye(3).tolist(),
            "depth_mm": 20.0,
            "opening_mm": 40.0,
        }
        converted = record_to_official_components(record, transform)
        np.testing.assert_allclose(converted["center_m"], [0.1, 0.01, 0.025])
        np.testing.assert_allclose(converted["grasp_point_m"], [0.1, 0.01, 0.005])

    def test_common_summary_does_not_re_rank_or_drop_ours_records(self):
        from scripts.common_eval.ours_official_common_eval import summarize_common_evaluation

        summary = summarize_common_evaluation(
            records=[{"score_total": 0.2}, {"score_total": 0.9}],
            collision=np.asarray([False, True]),
            empty=np.asarray([False, False]),
            mu_min=np.asarray([0.3, -1.0]),
            scored_mask=np.asarray([True, False]),
            error_mask=np.asarray([False, False]),
            wall_time_s=1.5,
        )

        self.assertEqual(summary["n_candidates"], 2)
        self.assertEqual(summary["n_geometry_valid"], 1)
        self.assertEqual(summary["common_fc_valid"], 1)
        self.assertEqual(summary["common_eval_count"], 2)
        self.assertEqual(summary["common_fc_valid_rate"], 0.5)
        self.assertEqual(summary["common_fc_valid_rate_geometry"], 1.0)
        self.assertEqual(summary["native_score_order_preserved"], True)
        self.assertEqual(summary["common_eval_wall_time_s"], 1.5)

    def test_common_summary_exposes_raw_and_unique_yields_separately(self):
        from scripts.common_eval.ours_official_common_eval import summarize_common_evaluation

        summary = summarize_common_evaluation(
            records=[{"score_total": 0.2}, {"score_total": 0.9}],
            collision=np.asarray([False, False]),
            empty=np.asarray([False, False]),
            mu_min=np.asarray([0.2, 0.8]),
            scored_mask=np.asarray([True, True]),
            error_mask=np.asarray([False, False]),
            wall_time_s=1.0,
            native_raw_count=10,
            native_unique_count=2,
        )
        self.assertEqual(summary["n_raw_candidates"], 10)
        self.assertEqual(summary["n_unique_outputs"], 2)
        self.assertAlmostEqual(summary["fc_yield_raw"], 0.2)
        self.assertAlmostEqual(summary["hq_yield_raw"], 0.1)
        self.assertAlmostEqual(summary["hq_rate_among_fc"], 0.5)


if __name__ == "__main__":
    unittest.main()
