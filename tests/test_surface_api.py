import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from grasp_detect import grasp_detect_from_surface


class SurfaceApiTests(unittest.TestCase):
    def test_validates_surface_shapes_before_generation(self):
        with self.assertRaises(ValueError):
            grasp_detect_from_surface(None, [[0, 0, 0]], [], [0, 0, 1])
        with self.assertRaises(ValueError):
            grasp_detect_from_surface(None, [[0, 0, 0]], [[0, 0, 1]], [0, 0, 0])

    def test_selected_surface_changes_generation_origin(self):
        object_data = SimpleNamespace(
            center=np.zeros(3),
            sample_radius=200.0,
            obj_axes=np.eye(3),
            ply_path="object.ply",
        )
        surface_normals = np.array([[1.0, 0.0, 0.0]])
        captured_origins = []

        def capture(*args, **kwargs):
            captured_origins.append(kwargs["frame_override"]["origin"].copy())
            return None, [], [], [], []

        with patch("grasp_detect.grasp_detect", side_effect=capture):
            grasp_detect_from_surface(object_data, [[1, 0, 0]], surface_normals, [1, 0, 0])
            grasp_detect_from_surface(object_data, [[-1, 0, 0]], surface_normals, [-1, 0, 0])

        self.assertFalse(np.allclose(captured_origins[0], captured_origins[1]))


if __name__ == "__main__":
    unittest.main()
