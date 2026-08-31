import unittest

import numpy as np

from surface_visibility import select_front_facing_surface
from view_sampling import fibonacci_directions


class ViewSamplingVisibilityTests(unittest.TestCase):
    def test_fibonacci_directions_are_unit_vectors(self):
        directions = fibonacci_directions(20)
        self.assertEqual(directions.shape, (20, 3))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)

    def test_single_view_is_supported(self):
        np.testing.assert_allclose(fibonacci_directions(1), [[0.0, 0.0, 1.0]])

    def test_front_facing_filter_uses_center_to_camera_convention(self):
        points = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=float)
        normals = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=float)
        selected, _ = select_front_facing_surface(points, normals, [1, 0, 0])
        np.testing.assert_array_equal(selected, [[1, 0, 0]])

    def test_rejects_invalid_view_count(self):
        with self.assertRaises(ValueError):
            fibonacci_directions(0)


if __name__ == "__main__":
    unittest.main()
