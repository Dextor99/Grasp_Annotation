import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from grasp_pipeline import GraspAnnotationResult


def _record(score, translation):
    return {
        "translation": list(translation),
        "rotation_matrix": np.eye(3).tolist(),
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "opening_mm": 45.0,
        "depth_mm": 20.0,
        "score_total": score,
        "view_id": 0,
        "anchor_id": 0,
        "approach_id": 0,
        "view_direction": [0.0, 0.0, 1.0],
        "anchor_point": [0.0, 0.0, 0.0],
        "anchor_normal": [0.0, 0.0, 1.0],
        "approach_direction": [0.0, 0.0, 1.0],
        "source_view_ids": [0],
        "source_anchor_ids": [0],
        "source_approach_ids": [0],
        "source_ids": [{"view_id": 0, "anchor_id": 0, "approach_id": 0}],
    }


class FinalizationAcceptanceTests(unittest.TestCase):
    def _result(self):
        raw = [_record(0.9, (0.0, 0.0, 0.0)), _record(0.8, (1.0, 0.0, 0.0))]
        unique = [raw[0]]
        return GraspAnnotationResult(
            raw_grasps=raw,
            unique_grasps=unique,
            meta={
                "raw_grasp_count": 2,
                "unique_grasp_count": 1,
                "merge_reduction_ratio": 0.5,
            },
        )

    def test_accepts_finite_sorted_nonempty_reduced_results(self):
        from grasp_freeze_validation import assert_annotation_invariants

        assert_annotation_invariants(self._result())

    def test_rejects_nan_empty_and_non_reducing_results(self):
        from grasp_freeze_validation import AcceptanceFailure, assert_annotation_invariants

        invalid = self._result()
        invalid.unique_grasps[0]["score_total"] = np.nan
        with self.assertRaises(AcceptanceFailure):
            assert_annotation_invariants(invalid)

        empty = GraspAnnotationResult([], [], {"raw_grasp_count": 0, "unique_grasp_count": 0})
        with self.assertRaises(AcceptanceFailure):
            assert_annotation_invariants(empty)

        non_reducing = self._result()
        non_reducing.unique_grasps = list(non_reducing.raw_grasps) + [_record(0.7, (2.0, 0.0, 0.0))]
        non_reducing.meta["unique_grasp_count"] = 3
        with self.assertRaises(AcceptanceFailure):
            assert_annotation_invariants(non_reducing)

    def test_accepts_unique_count_equal_to_raw_count(self):
        from grasp_freeze_validation import assert_annotation_invariants

        result = self._result()
        result.unique_grasps = list(result.raw_grasps)
        result.meta["unique_grasp_count"] = len(result.unique_grasps)
        result.meta["merge_reduction_ratio"] = 0.0
        assert_annotation_invariants(result)

    def test_repeated_results_require_identical_counts_top_poses_and_scores(self):
        from grasp_freeze_validation import AcceptanceFailure, assert_repeated_results_equal

        reference = self._result()
        repeated = self._result()
        assert_repeated_results_equal(reference, repeated, top_k=10)

        repeated.unique_grasps[0]["translation"][0] = 1e-6
        with self.assertRaises(AcceptanceFailure):
            assert_repeated_results_equal(reference, repeated, top_k=10)

    def test_rotation_and_direction_geometry_is_checked(self):
        from grasp_freeze_validation import AcceptanceFailure, assert_annotation_invariants

        invalid = self._result()
        invalid.unique_grasps[0]["rotation_matrix"][0][0] = 2.0
        with self.assertRaises(AcceptanceFailure):
            assert_annotation_invariants(invalid)

    def test_real_case_runner_completes_repeats_before_reporting_gate_failure(self):
        from scripts.validate_grasp_freeze import ValidationCase, run_validation_case

        invalid = self._result()
        invalid.unique_grasps = list(invalid.raw_grasps)
        invalid.meta.update(
            unique_grasp_count=2,
            timings={"total_s": 0.1},
        )
        case = ValidationCase("fixture", "fixture.ply", 3, 2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = Path(temporary_directory) / "fixture.ply"
            model.write_text("fixture", encoding="utf-8")
            with patch(
                "scripts.validate_grasp_freeze.run_grasp_annotation",
                return_value=invalid,
            ) as run, patch("scripts.validate_grasp_freeze.export_grasp_annotations"):
                summary = run_validation_case(
                    case,
                    temporary_directory,
                    temporary_directory,
                    repeats=2,
                )

        self.assertEqual(run.call_count, 2)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(len(summary["repeats"]), 2)
        self.assertTrue(summary["failures"])


if __name__ == "__main__":
    unittest.main()
