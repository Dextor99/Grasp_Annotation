import unittest

import numpy as np

from surface_anchor import (
    build_surface_anchors,
    estimate_local_normal,
    farthest_point_sample_indices,
)


class SurfaceAnchorTests(unittest.TestCase):
    def test_fps_is_deterministic_and_spatially_separated(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float)
        first = farthest_point_sample_indices(points, 2)
        second = farthest_point_sample_indices(points, 2)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first.tolist()), {0, 4})

    def test_fps_does_not_repeat_indices_for_duplicate_points(self):
        points = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        selected = farthest_point_sample_indices(points, num_samples=3)
        self.assertEqual(len(selected), len(set(selected.tolist())))

    def test_local_normal_is_smoothed_and_faces_view(self):
        points = np.array([[0, 0, 0], [0.1, 0, 0], [-0.1, 0, 0]], dtype=float)
        normals = np.array([[0, 0, -1], [0, 0.1, -0.995], [0, -0.1, -0.995]], dtype=float)
        normal = estimate_local_normal(points, normals, points[0], [0, 0, 1], normal_knn=3)
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0)
        self.assertGreater(float(normal @ np.array([0, 0, 1])), 0.0)

    def test_build_surface_anchors_returns_requested_count_and_ids(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        normals = np.tile([0, 1, 0], (4, 1))
        anchors = build_surface_anchors(points, normals, [0, 1, 0], num_anchors=3, normal_knn=2)
        self.assertEqual([anchor.anchor_id for anchor in anchors], [0, 1, 2])
        self.assertEqual(len({tuple(anchor.point) for anchor in anchors}), 3)

    def test_rejects_empty_surface(self):
        with self.assertRaises(ValueError):
            build_surface_anchors(np.empty((0, 3)), np.empty((0, 3)), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
