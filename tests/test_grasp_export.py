import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class GraspExportTests(unittest.TestCase):
    def _record(self):
        return {
            "translation": [1.0, 2.0, 3.0],
            "rotation_matrix": np.eye(3).tolist(),
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "opening_mm": 45.0,
            "depth_mm": 20.0,
            "score_total": 0.8,
            "view_id": 1,
            "anchor_id": 2,
            "approach_id": 3,
            "view_direction": [0.0, 0.0, 1.0],
            "anchor_point": [1.0, 2.0, 3.0],
            "anchor_normal": [0.0, 0.0, 1.0],
            "approach_direction": [0.0, 0.0, 1.0],
            "source_view_ids": [1],
            "source_anchor_ids": [2],
            "source_approach_ids": [3],
            "source_ids": [{"view_id": 1, "anchor_id": 2, "approach_id": 3}],
        }

    def test_writes_only_json_npz_and_meta_with_dense_numeric_arrays(self):
        from grasp_export import export_grasp_annotations
        from grasp_pipeline import GraspAnnotationResult

        result = GraspAnnotationResult(
            raw_grasps=[self._record(), self._record()],
            unique_grasps=[self._record()],
            meta={"raw_grasp_count": 2, "unique_grasp_count": 1, "config": {"mode": "cone"}},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "annotation"
            paths = export_grasp_annotations(result, output)

            self.assertEqual({path.name for path in output.iterdir()}, {"grasps.json", "grasps.npz", "meta.json"})
            self.assertEqual(set(paths), {"grasps_json", "grasps_npz", "meta_json"})
            grasps = json.loads((output / "grasps.json").read_text(encoding="utf-8"))
            meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
            with np.load(output / "grasps.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["translations"].shape, (1, 3))
                self.assertEqual(arrays["rotation_matrices"].shape, (1, 3, 3))
                self.assertEqual(arrays["quaternions_xyzw"].shape, (1, 4))
                self.assertEqual(arrays["scores_total"].shape, (1,))
                self.assertEqual(arrays["search_openings_mm"].shape, (1,))
                self.assertEqual(arrays["grasp_widths_mm"].shape, (1,))
                self.assertEqual(arrays["support_spans_mm"].shape, (1,))
                self.assertEqual(arrays["closure_center_offsets_mm"].shape, (1,))
                self.assertEqual(arrays["effective_margins_mm"].shape, (1,))
                self.assertTrue(all(np.issubdtype(array.dtype, np.number) for array in arrays.values()))

            self.assertEqual(len(grasps), 1)
            self.assertEqual(meta["raw_grasp_count"], 2)

    def test_empty_results_keep_stable_npz_shapes(self):
        from grasp_export import export_grasp_annotations
        from grasp_pipeline import GraspAnnotationResult

        result = GraspAnnotationResult(raw_grasps=[], unique_grasps=[], meta={})
        with tempfile.TemporaryDirectory() as temporary_directory:
            export_grasp_annotations(result, temporary_directory)
            with np.load(Path(temporary_directory) / "grasps.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["translations"].shape, (0, 3))
                self.assertEqual(arrays["rotation_matrices"].shape, (0, 3, 3))
                self.assertEqual(arrays["quaternions_xyzw"].shape, (0, 4))

    def test_rejects_unexpected_existing_files_without_deleting_them(self):
        from grasp_export import export_grasp_annotations
        from grasp_pipeline import GraspAnnotationResult

        result = GraspAnnotationResult(raw_grasps=[], unique_grasps=[], meta={})
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            extra = output / "keep-me.csv"
            extra.write_text("user data", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                export_grasp_annotations(result, output)

            self.assertEqual(extra.read_text(encoding="utf-8"), "user data")
            self.assertEqual({path.name for path in output.iterdir()}, {"keep-me.csv"})

    def test_validation_failure_does_not_leave_partial_final_files(self):
        from grasp_export import export_grasp_annotations
        from grasp_pipeline import GraspAnnotationResult

        invalid = self._record()
        invalid["score_total"] = np.nan
        result = GraspAnnotationResult(raw_grasps=[], unique_grasps=[invalid], meta={})
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            with self.assertRaises(ValueError):
                export_grasp_annotations(result, output)
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
