import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class GraspNetExecutionProtocolTests(unittest.TestCase):
    def test_runner_requires_sdf_prefix_before_loading_mesh(self):
        from baselines.graspnet_annotation.run_graspnet_baseline import run

        with self.assertRaisesRegex(ValueError, "sdf_prefix"):
            run(Path("does-not-exist.obj"), Path("out"), input_unit="m")

    def test_geometry_runner_handles_multiple_grasp_points(self):
        import trimesh

        from baselines.graspnet_annotation.run_graspnet_baseline import run

        mesh = Path(__file__).parents[2] / "baselines" / "graspnet_annotation" / "assets" / "debug_cube" / "debug_cube.obj"
        with tempfile.TemporaryDirectory() as temporary_directory:
            fake_views = np.zeros((300, 3), dtype=np.float32)
            fake_views[:, 2] = 1.0
            fake_rotations = lambda views, angles: np.repeat(np.eye(3, dtype=np.float32)[None, ...], len(views), axis=0)
            with patch("baselines.graspnet_annotation.candidate_generation.generate_views", return_value=fake_views), \
                    patch("baselines.graspnet_annotation.candidate_generation.generate_view_rotations", side_effect=fake_rotations), \
                    patch("baselines.graspnet_annotation.label_arrays.generate_views", return_value=fake_views), \
                    patch("baselines.graspnet_annotation.label_arrays.generate_view_rotations", side_effect=fake_rotations):
                summary = run(mesh, Path(temporary_directory) / "run", input_unit="m", max_points=2, skip_force_closure=True)
        self.assertEqual(summary["n_grasp_points"], 2)
        self.assertEqual(summary["n_candidates"], 2 * 300 * 12 * 4)

    def test_formal_sdf_default_is_100(self):
        from scripts.baselines.generate_sdf import DEFAULT_GRID_DIM

        self.assertEqual(DEFAULT_GRID_DIM, 100)

    def test_shard_executor_rejects_nonpositive_worker_count(self):
        from scripts.baselines.run_force_closure_shards import run

        with self.assertRaisesRegex(ValueError, "workers"):
            run(Path("does-not-exist"), Path("does-not-exist.sdf"), Path("shards"), workers=0)

    def test_merge_rejects_inconsistent_shard_lengths(self):
        from baselines.graspnet_annotation.config import DenseAnnotationConfig
        from baselines.graspnet_annotation.label_arrays import RawLabelArrays
        from scripts.baselines.merge_force_closure_shards import merge

        config = DenseAnnotationConfig.full(num_views=1, num_angles=1, depths_m=(0.01, 0.02))
        labels = RawLabelArrays.create(1, config)
        labels.collision[:] = False
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            geometry = root / "geometry"
            shards = root / "shards"
            geometry.mkdir()
            shards.mkdir()
            np.savez_compressed(
                geometry / "grasp_labels.npz",
                points=labels.points,
                offsets=labels.offsets,
                collision=labels.collision,
                scores=labels.scores,
            )
            # candidate_ids has two entries but masks/scores have one.
            np.savez_compressed(
                shards / "fc_shard_000000.npz",
                candidate_ids=np.array([0, 1], dtype=np.int64),
                mu_min=np.array([0.1], dtype=np.float32),
                scored_mask=np.array([True], dtype=bool),
                error_mask=np.array([False], dtype=bool),
                elapsed_s=np.array(0.1, dtype=np.float64),
            )
            with self.assertRaisesRegex(RuntimeError, "length"):
                merge(geometry, shards, root / "merged")

    def test_merge_rejects_nonfinite_scored_value(self):
        from baselines.graspnet_annotation.config import DenseAnnotationConfig
        from baselines.graspnet_annotation.label_arrays import RawLabelArrays
        from scripts.baselines.merge_force_closure_shards import merge

        config = DenseAnnotationConfig.full(num_views=1, num_angles=1, depths_m=(0.01,))
        labels = RawLabelArrays.create(1, config)
        labels.collision[:] = False
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            geometry = root / "geometry"
            shards = root / "shards"
            geometry.mkdir()
            shards.mkdir()
            np.savez_compressed(
                geometry / "grasp_labels.npz",
                points=labels.points,
                offsets=labels.offsets,
                collision=labels.collision,
                scores=labels.scores,
            )
            np.savez_compressed(
                shards / "fc_shard_000000.npz",
                candidate_ids=np.array([0], dtype=np.int64),
                mu_min=np.array([np.nan], dtype=np.float32),
                scored_mask=np.array([True], dtype=bool),
                error_mask=np.array([False], dtype=bool),
                elapsed_s=np.array(0.1, dtype=np.float64),
            )
            with self.assertRaisesRegex(RuntimeError, "finite"):
                merge(geometry, shards, root / "merged")

    def test_merge_exports_full_audit_masks(self):
        from baselines.graspnet_annotation.config import DenseAnnotationConfig
        from baselines.graspnet_annotation.label_arrays import RawLabelArrays
        from scripts.baselines.merge_force_closure_shards import merge

        config = DenseAnnotationConfig.full(num_views=1, num_angles=1, depths_m=(0.01, 0.02))
        labels = RawLabelArrays.create(1, config)
        labels.collision[:] = False
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            geometry = root / "geometry"
            shards = root / "shards"
            geometry.mkdir()
            shards.mkdir()
            np.savez_compressed(
                geometry / "grasp_labels.npz",
                points=labels.points,
                offsets=labels.offsets,
                collision=labels.collision,
                scores=labels.scores,
            )
            np.savez_compressed(
                shards / "fc_shard_000000.npz",
                candidate_ids=np.array([0, 1], dtype=np.int64),
                mu_min=np.array([0.1, 0.2], dtype=np.float32),
                scored_mask=np.array([True, True], dtype=bool),
                error_mask=np.array([False, False], dtype=bool),
                elapsed_s=np.array(0.1, dtype=np.float64),
            )
            merge(geometry, shards, root / "merged")
            self.assertTrue((root / "merged" / "scored_mask.npy").is_file())
            self.assertTrue((root / "merged" / "error_mask.npy").is_file())


if __name__ == "__main__":
    unittest.main()
