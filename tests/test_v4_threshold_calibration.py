import csv
import json
import tempfile
import unittest
from pathlib import Path


class V4ThresholdCalibrationTests(unittest.TestCase):
    def test_calibration_ignores_uncertain_and_finds_balanced_cutoff(self):
        from scripts.calibrate_v4_threshold import calibrate_rows

        labels = [
            {"sample_id": "1", "object": "a", "human_label": "G"},
            {"sample_id": "2", "object": "a", "human_label": "B"},
            {"sample_id": "3", "object": "a", "human_label": "U"},
            {"sample_id": "4", "object": "b", "human_label": "G"},
            {"sample_id": "5", "object": "b", "human_label": "B"},
        ]
        manifest = [
            {"sample_id": "1", "score_total_v4": "0.9"},
            {"sample_id": "2", "score_total_v4": "0.1"},
            {"sample_id": "3", "score_total_v4": "0.5"},
            {"sample_id": "4", "score_total_v4": "0.8"},
            {"sample_id": "5", "score_total_v4": "0.2"},
        ]

        result = calibrate_rows(labels, manifest)

        self.assertGreaterEqual(result["best_threshold"], 0.2)
        self.assertLessEqual(result["best_threshold"], 0.8)
        self.assertEqual(result["labeled_count"], 4)
        self.assertEqual(result["uncertain_count"], 1)
        self.assertGreater(result["best_balanced_accuracy"], 0.9)
        self.assertEqual(len(result["leave_one_object_out"]), 2)

    def test_report_contains_only_json_serializable_scalars(self):
        from scripts.calibrate_v4_threshold import calibrate_rows

        labels = [
            {"sample_id": "1", "object": "a", "human_label": "G"},
            {"sample_id": "2", "object": "a", "human_label": "B"},
            {"sample_id": "3", "object": "b", "human_label": "G"},
            {"sample_id": "4", "object": "b", "human_label": "B"},
        ]
        manifest = [
            {"sample_id": "1", "score_total_v4": "0.9"},
            {"sample_id": "2", "score_total_v4": "0.1"},
            {"sample_id": "3", "score_total_v4": "0.8"},
            {"sample_id": "4", "score_total_v4": "0.2"},
        ]

        report = calibrate_rows(labels, manifest)

        # Regression guard for NumPy scalar leakage from np.linspace and
        # boolean aggregations (the original failure was int32 in JSON).
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
