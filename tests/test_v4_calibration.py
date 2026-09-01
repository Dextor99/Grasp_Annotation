import csv
import tempfile
import unittest
from pathlib import Path


class V4CalibrationTests(unittest.TestCase):
    def test_template_selects_top_and_bottom_without_assigning_labels(self):
        from scripts.create_v4_calibration_set import build_calibration_rows

        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            records = [{"score_total": score, "score_total_v4": score} for score in (0.1, 0.2, 0.3, 0.4)]
            (result_dir / "grasps.json").write_text(__import__("json").dumps(records), encoding="utf-8")
            rows = build_calibration_rows([("toy", result_dir)], per_group=2)

        self.assertEqual(len(rows), 4)
        self.assertEqual([row["selection_group"] for row in rows], ["top_candidate", "top_candidate", "bottom_candidate", "bottom_candidate"])
        self.assertTrue(all(row["human_label"] == "" for row in rows))
        self.assertEqual(rows[0]["record_index"], 3)
        self.assertEqual(rows[-1]["record_index"], 0)


if __name__ == "__main__":
    unittest.main()
