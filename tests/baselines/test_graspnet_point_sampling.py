import unittest
from pathlib import Path

import numpy as np


class GraspNetPointSamplingTests(unittest.TestCase):
    def test_grasp_and_collision_clouds_use_distinct_voxel_contracts(self):
        import trimesh

        from baselines.graspnet_annotation.config import DenseAnnotationConfig
        from baselines.graspnet_annotation.grasp_point_sampling import sample_collision_points, sample_grasp_points

        mesh = trimesh.load_mesh(
            Path(__file__).parents[2] / "baselines/graspnet_annotation/assets/debug_cube/debug_cube.obj",
            process=False,
        )
        config = DenseAnnotationConfig.full(surface_samples=300, max_grasp_points=20)
        grasp = sample_grasp_points(mesh, config)
        collision = sample_collision_points(mesh, config)
        self.assertLessEqual(len(grasp), 20)
        self.assertGreater(len(collision), 0)
        self.assertEqual(grasp.dtype, np.float32)
        self.assertEqual(collision.dtype, np.float32)

    def test_sampling_is_deterministic_for_same_seed(self):
        import trimesh

        from baselines.graspnet_annotation.config import DenseAnnotationConfig
        from baselines.graspnet_annotation.grasp_point_sampling import sample_collision_points, sample_grasp_points

        mesh = trimesh.load_mesh(
            Path(__file__).parents[2] / "baselines/graspnet_annotation/assets/debug_cube/debug_cube.obj",
            process=False,
        )
        config = DenseAnnotationConfig.full(surface_samples=200, max_grasp_points=10)
        np.testing.assert_array_equal(sample_grasp_points(mesh, config), sample_grasp_points(mesh, config))
        np.testing.assert_array_equal(sample_collision_points(mesh, config), sample_collision_points(mesh, config))


if __name__ == "__main__":
    unittest.main()
