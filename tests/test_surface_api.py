import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from grasp_detect import (
    _resolve_contact_geometry,
    grasp_detect_from_anchor_approach,
    grasp_detect_from_surface,
)


class SurfaceApiTests(unittest.TestCase):
    def test_anchor_contact_geometry_skips_cylinder_search(self):
        class SpyProfiler:
            def measure(self, *args, **kwargs):
                raise AssertionError("anchor contact must not run cylinder search")

        frame = {"z_axis": np.array([0.0, 0.0, -1.0])}
        cyl0, cyl1, center0, center1 = _resolve_contact_geometry(
            SpyProfiler(),
            origin=np.zeros(3),
            points=np.zeros((1, 3)),
            frame=frame,
            contact_center_override=np.array([1.0, 2.0, 3.0]),
        )
        self.assertIsNone(cyl0)
        self.assertIsNone(cyl1)
        np.testing.assert_allclose(center0, [1, 2, 3])
        np.testing.assert_allclose(center1, [1, 2, -37])

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

    def test_anchor_approach_builds_expected_frame_and_metadata(self):
        object_data = SimpleNamespace(ply_path="object.ply")
        captured_frames = []
        captured_objects = []
        captured_contacts = []

        def capture(*args, **kwargs):
            captured_frames.append(kwargs["frame_override"])
            captured_objects.append(kwargs["object_data"])
            captured_contacts.append(kwargs["contact_center_override"])
            return None, [{"id": 7}], [], [], []

        with patch("grasp_detect.grasp_detect", side_effect=capture):
            grasps = grasp_detect_from_anchor_approach(
                object_data,
                anchor_point=[1, 2, 3],
                approach_direction=[0, 0, -1],
                anchor_normal=[0, 0, 1],
                metadata={"anchor_id": 2, "approach_id": 4},
            )

        np.testing.assert_allclose(captured_frames[0]["origin"], [1, 2, 3])
        np.testing.assert_allclose(captured_frames[0]["z_axis"], [0, 0, -1])
        self.assertIs(captured_objects[0], object_data)
        np.testing.assert_allclose(captured_contacts[0], [1, 2, 3])
        self.assertEqual(grasps[0]["anchor_id"], 2)
        self.assertEqual(grasps[0]["approach_id"], 4)
        np.testing.assert_allclose(grasps[0]["anchor_point"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
