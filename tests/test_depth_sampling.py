import unittest

import numpy as np

from depth_sampling import generate_depth_samples


class DepthSamplingTests(unittest.TestCase):
    def test_depths_scale_with_object_radius(self):
        depths = generate_depth_samples(object_radius=100.0, num_depth=4, max_ratio=1.2)
        np.testing.assert_allclose(depths, [0.0, 40.0, 80.0, 120.0])

    def test_depth_count_and_monotonicity_are_stable(self):
        depths = generate_depth_samples(object_radius=117.4398, num_depth=16, max_ratio=1.2)
        self.assertEqual(len(depths), 16)
        self.assertEqual(depths[0], 0.0)
        self.assertTrue(np.all(np.diff(depths) > 0))

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            generate_depth_samples(object_radius=0.0)
        with self.assertRaises(ValueError):
            generate_depth_samples(object_radius=100.0, num_depth=1)
        with self.assertRaises(ValueError):
            generate_depth_samples(object_radius=100.0, max_ratio=0.0)


if __name__ == "__main__":
    unittest.main()
