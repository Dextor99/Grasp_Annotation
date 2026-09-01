import json
import tempfile
import unittest
from pathlib import Path


class GraspQualityAnalysisTests(unittest.TestCase):
    def test_computes_threshold_counts_and_ratios(self):
        from scripts.analyze_grasp_quality import compute_score_statistics

        stats = compute_score_statistics(
            [{"score_total": value} for value in (-0.2, 0.2, 0.8, 0.95, 1.0)]
        )

        self.assertEqual(stats["unique_grasp_count"], 5)
        self.assertEqual(stats["score_ge_0_count"], 4)
        self.assertAlmostEqual(stats["score_ge_0_ratio"], 0.8)
        self.assertEqual(stats["score_ge_0_8_count"], 3)
        self.assertAlmostEqual(stats["score_ge_0_95_ratio"], 0.4)
        self.assertAlmostEqual(stats["score_median"], 0.8)

    def test_analyzes_result_csv_and_writes_output(self):
        from scripts.analyze_grasp_quality import analyze_csv

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result_dir = root / "cat"
            result_dir.mkdir()
            (result_dir / "grasps.json").write_text(
                json.dumps([{"score_total": 0.9}, {"score_total": 0.1}]),
                encoding="utf-8",
            )
            input_csv = root / "input.csv"
            input_csv.write_text(
                "result_dir,object,generated_candidate_count\n"
                f"{result_dir.as_posix()},model/cat.ply,10\n",
                encoding="utf-8",
            )
            output_csv = root / "quality.csv"

            rows = analyze_csv(input_csv, output_csv)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["unique_grasp_count"], 2)
            self.assertAlmostEqual(rows[0]["high_quality_yield"], 0.1)
            self.assertAlmostEqual(rows[0]["high_quality_threshold"], 0.13)
            self.assertTrue(output_csv.is_file())

    def test_writes_weighted_aggregate_statistics(self):
        from scripts.analyze_grasp_quality import analyze_csv

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result_dir = root / "object"
            result_dir.mkdir()
            (result_dir / "grasps.json").write_text(
                json.dumps([{"score_total": 0.0}, {"score_total": 1.0}]),
                encoding="utf-8",
            )
            input_csv = root / "input.csv"
            input_csv.write_text(
                "result_dir,object,generated_candidate_count\n"
                f"{result_dir.as_posix()},model/object.ply,10\n",
                encoding="utf-8",
            )
            output_csv = root / "quality.csv"
            aggregate_csv = root / "aggregate.csv"

            analyze_csv(input_csv, output_csv, aggregate_csv)

            with aggregate_csv.open(encoding="utf-8", newline="") as handle:
                aggregate = list(__import__("csv").DictReader(handle))[0]
            self.assertEqual(aggregate["unique_grasp_count"], "2")
            self.assertEqual(aggregate["score_ge_0_8_count"], "1")
            self.assertEqual(aggregate["generated_candidate_count"], "10")
            self.assertEqual(aggregate["high_quality_yield"], "0.1")


if __name__ == "__main__":
    unittest.main()
