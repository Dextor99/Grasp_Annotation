import json
import tempfile
import unittest
from pathlib import Path


class ExperimentSummaryTests(unittest.TestCase):
    def test_summarizes_explicit_candidate_counts_and_rates(self):
        from scripts.summarize_grasp_results import summarize_result_directory

        with tempfile.TemporaryDirectory() as temporary_directory:
            result_dir = Path(temporary_directory) / "cat_v1"
            result_dir.mkdir()
            (result_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "object": "model/cat.ply",
                        "raw_grasp_count": 8,
                        "unique_grasp_count": 4,
                        "candidate_counts": {
                            "generated_candidate_count": 20,
                            "scored_candidate_count": 20,
                            "refinement_input_count": 20,
                            "closure_geometry_rejected": 3,
                            "closure_pose_collision_rejected": 1,
                            "closure_valid_count": 16,
                            "unique_grasp_count": 4,
                        },
                        "timings": {"total_s": 2.5},
                    }
                ),
                encoding="utf-8",
            )
            (result_dir / "grasps.json").write_text(
                json.dumps([{"score_total": 0.9}, {"score_total": 0.7}, {"score_total": 0.5}]),
                encoding="utf-8",
            )

            summary = summarize_result_directory(result_dir)

        self.assertEqual(summary["generated_candidate_count"], 20)
        self.assertEqual(summary["closure_valid_count"], 16)
        self.assertAlmostEqual(summary["closure_acceptance_rate"], 0.8)
        self.assertAlmostEqual(summary["merge_retention_rate"], 0.25)
        self.assertAlmostEqual(summary["mean_score"], 0.7)
        self.assertAlmostEqual(summary["top1_score"], 0.9)
        self.assertAlmostEqual(summary["top20_mean_score"], 0.7)
        self.assertEqual(summary["high_quality_count"], 1)
        self.assertAlmostEqual(summary["high_quality_ratio"], 1 / 3)
        self.assertAlmostEqual(summary["high_quality_yield"], 1 / 20)
        self.assertEqual(summary["total_s"], 2.5)

    def test_legacy_metadata_falls_back_to_raw_count(self):
        from scripts.summarize_grasp_results import summarize_result_directory

        with tempfile.TemporaryDirectory() as temporary_directory:
            result_dir = Path(temporary_directory)
            (result_dir / "meta.json").write_text(
                json.dumps({"raw_grasp_count": 3, "unique_grasp_count": 2}),
                encoding="utf-8",
            )
            summary = summarize_result_directory(result_dir)

        self.assertEqual(summary["generated_candidate_count"], 3)
        self.assertEqual(summary["closure_valid_count"], 3)
        self.assertAlmostEqual(summary["merge_retention_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
