import unittest

import numpy as np

from surface_visibility import filter_front_facing_surface


class FilterFrontFacingSurfaceTests(unittest.TestCase):
    def test_removes_back_facing_and_zero_normal_points(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])

        visible_points, visible_normals, mask = filter_front_facing_surface(
            points, normals, [0.0, 0.0, 1.0]
        )

        np.testing.assert_array_equal(mask, [True, False, False])
        np.testing.assert_array_equal(visible_points, points[:1])
        np.testing.assert_allclose(visible_normals, [[0.0, 0.0, 1.0]])

    def test_rejects_mismatched_point_and_normal_counts(self):
        with self.assertRaises(ValueError):
            filter_front_facing_surface(
                np.zeros((2, 3)), np.zeros((1, 3)), [0.0, 0.0, 1.0]
            )

    def test_rejects_a_zero_view_direction(self):
        with self.assertRaises(ValueError):
            filter_front_facing_surface(
                np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), [0.0, 0.0, 0.0]
            )

    def test_retains_aligned_extreme_magnitude_vectors(self):
        points = np.array([[0.0, 0.0, 0.0]])
        for magnitude in (1e308, 1e-308):
            with self.subTest(magnitude=magnitude):
                visible_points, visible_normals, mask = filter_front_facing_surface(
                    points,
                    np.array([[magnitude, 0.0, 0.0]]),
                    [magnitude, 0.0, 0.0],
                )

                np.testing.assert_array_equal(mask, [True])
                np.testing.assert_array_equal(visible_points, points)
                np.testing.assert_allclose(visible_normals, [[1.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
