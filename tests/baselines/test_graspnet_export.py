import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class GraspNetExportTests(unittest.TestCase):
    def test_raw_label_arrays_have_official_axes_and_invalid_score_sentinel(self):
        from baselines.graspnet_annotation.config import DenseAnnotationConfig
        from baselines.graspnet_annotation.label_arrays import RawLabelArrays

        labels = RawLabelArrays.create(point_count=1, config=DenseAnnotationConfig.full())
        self.assertEqual(labels.points.shape, (1, 3))
        self.assertEqual(labels.offsets.shape, (1, 300, 12, 4, 3))
        self.assertEqual(labels.collision.shape, (1, 300, 12, 4))
        self.assertTrue(np.all(labels.scores == -1.0))

    def test_valid_grasp_conversion_has_documented_17_columns(self):
        from baselines.graspnet_annotation.config import DenseAnnotationConfig
        from baselines.graspnet_annotation.label_arrays import RawLabelArrays

        config = DenseAnnotationConfig.full(num_views=1, num_angles=1, depths_m=(0.01,))
        labels = RawLabelArrays.create(point_count=1, config=config)
        labels.points[0] = [0.1, 0.2, 0.3]
        labels.collision[0, 0, 0, 0] = False
        labels.scores[0, 0, 0, 0] = 0.3
        labels.offsets[0, 0, 0, 0, 2] = 0.04
        with patch("baselines.graspnet_annotation.label_arrays.generate_views", return_value=np.array([[0.0, 0.0, 1.0]], dtype=np.float32)), patch(
            "baselines.graspnet_annotation.label_arrays.generate_view_rotations", return_value=np.eye(3, dtype=np.float32)[None, ...]
        ):
            valid = labels.to_valid_grasps(config)
        self.assertEqual(valid.shape, (1, 17))
        self.assertAlmostEqual(float(valid[0, 0]), 0.8)
        self.assertAlmostEqual(float(valid[0, 1]), 0.04)

    def test_export_refuses_to_mix_with_existing_user_files(self):
        from baselines.graspnet_annotation.config import DenseAnnotationConfig
        from baselines.graspnet_annotation.export import export_annotation_run
        from baselines.graspnet_annotation.label_arrays import RawLabelArrays

        labels = RawLabelArrays.create(1, DenseAnnotationConfig.full(num_views=1, num_angles=1, depths_m=(0.01,)))
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run"
            output.mkdir()
            (output / "keep.txt").write_text("user file", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_annotation_run(output, labels, {"units": "m"}, [])
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "user file")

    def test_export_writes_five_documented_files(self):
        from baselines.graspnet_annotation.config import DenseAnnotationConfig
        from baselines.graspnet_annotation.export import export_annotation_run
        from baselines.graspnet_annotation.label_arrays import RawLabelArrays

        config = DenseAnnotationConfig.full(num_views=1, num_angles=1, depths_m=(0.01,))
        labels = RawLabelArrays.create(1, config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run"
            export_annotation_run(output, labels, {"units": "m"}, [{"stage": "test", "seconds": 0.1}])
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"grasp_labels.npz", "valid_grasps.npy", "summary.json", "timing.csv", "run_config.json"},
            )

    def test_export_accepts_an_existing_empty_directory(self):
        from baselines.graspnet_annotation.config import DenseAnnotationConfig
        from baselines.graspnet_annotation.export import export_annotation_run
        from baselines.graspnet_annotation.label_arrays import RawLabelArrays

        config = DenseAnnotationConfig.full(num_views=1, num_angles=1, depths_m=(0.01,))
        labels = RawLabelArrays.create(1, config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run"
            output.mkdir()
            export_annotation_run(output, labels, {"units": "m"}, [])
            self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
