import random
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d


class GraspDeterminismTests(unittest.TestCase):
    def test_reseeding_repeats_python_and_numpy_sequences(self):
        from grasp_determinism import configure_determinism

        configure_determinism(True, seed=17)
        first = (random.random(), np.random.random())
        configure_determinism(True, seed=17)
        second = (random.random(), np.random.random())

        self.assertEqual(first, second)

    def test_seeds_open3d_and_disabled_mode_is_noop(self):
        from grasp_determinism import configure_determinism

        with patch("grasp_determinism.o3d.utility.random.seed") as seed_open3d:
            configure_determinism(True, seed=23)
            seed_open3d.assert_called_once_with(23)
            seed_open3d.reset_mock()
            configure_determinism(False, seed=23)
            seed_open3d.assert_not_called()

    def test_open3d_surface_sampling_repeats_after_reseeding(self):
        from grasp_determinism import configure_determinism

        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=2.0, depth=3.0)
        configure_determinism(True, seed=31)
        first = np.asarray(mesh.sample_points_uniformly(number_of_points=64).points)
        configure_determinism(True, seed=31)
        second = np.asarray(mesh.sample_points_uniformly(number_of_points=64).points)

        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
