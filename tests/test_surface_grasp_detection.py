import inspect
import unittest
from unittest.mock import patch

import numpy as np

import grasp_detect


class SurfaceGraspDetectionTests(unittest.TestCase):
    def test_legacy_detector_keeps_two_parameter_signature(self):
        self.assertEqual(
            list(inspect.signature(grasp_detect.grasp_detect).parameters),
            ["ply_path", "i"],
        )

    def test_rejects_invalid_surface_inputs(self):
        invalid_cases = [
            (np.zeros((2, 3)), np.zeros((1, 3)), [0.0, 0.0, 1.0]),
            (np.zeros((2, 2)), np.zeros((2, 3)), [0.0, 0.0, 1.0]),
            (np.array([[np.nan, 0.0, 0.0]]), np.zeros((1, 3)), [0.0, 0.0, 1.0]),
            (np.zeros((1, 3)), np.zeros((1, 3)), [0.0, 0.0, 0.0]),
            (np.zeros((1, 3)), np.zeros((1, 3)), [np.inf, 0.0, 0.0]),
        ]

        for points, normals, view in invalid_cases:
            with self.subTest(points=points, normals=normals, view=view):
                with self.assertRaises(ValueError):
                    grasp_detect.grasp_detect_from_surface(points, normals, view)

    def test_empty_surface_returns_no_candidates(self):
        result = grasp_detect.grasp_detect_from_surface(
            np.empty((0, 3)), np.empty((0, 3)), [0.0, 0.0, 1.0]
        )

        self.assertEqual(result, [])

    def test_accepts_a_nonzero_view_direction_that_would_underflow_norm(self):
        result = grasp_detect.grasp_detect_from_surface(
            np.empty((0, 3)), np.empty((0, 3)), [1e-308, 0.0, 0.0]
        )

        self.assertEqual(result, [])

    def test_packages_candidates_with_metadata_and_provenance(self):
        candidate = {
            "T_gripper_object": np.eye(4),
            "opening": 45,
        }
        metadata = {"object_id": "mug-7"}
        with patch.object(grasp_detect, "_generate_surface_candidates", return_value=[candidate]) as core:
            result = grasp_detect.grasp_detect_from_surface(
                [[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [0.0, 0.0, 2.0], metadata
            )

        core.assert_called_once()
        self.assertEqual(result[0]["opening"], 45)
        np.testing.assert_allclose(result[0]["T_gripper_object"], np.eye(4))
        np.testing.assert_allclose(result[0]["view_direction"], [0.0, 0.0, 1.0])
        self.assertEqual(result[0]["object_id"], "mug-7")
        self.assertEqual(metadata, {"object_id": "mug-7"})

    def test_returns_no_candidates_when_cylinder_has_no_valid_contact(self):
        with patch.object(
            grasp_detect,
            "generate_cylinder_sections",
            return_value=(None, None, None, None),
        ):
            result = grasp_detect.grasp_detect_from_surface(
                [[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [0.0, 0.0, 1.0]
            )

        self.assertEqual(result, [])

    def test_collision_treats_empty_gripper_samples_as_invalid(self):
        empty_mesh = grasp_detect.o3d.geometry.TriangleMesh()
        point_cloud = grasp_detect.o3d.geometry.PointCloud()
        point_cloud.points = grasp_detect.o3d.utility.Vector3dVector([[0.0, 0.0, 0.0]])

        sampled_points = grasp_detect.o3d.geometry.PointCloud()
        with patch.object(
            grasp_detect.o3d.geometry.TriangleMesh,
            "sample_points_uniformly",
            return_value=sampled_points,
        ):
            self.assertTrue(grasp_detect.check_collision([empty_mesh, empty_mesh, empty_mesh], point_cloud))

    def test_legacy_detector_returns_empty_candidates_when_contact_is_invalid(self):
        point_cloud = grasp_detect.o3d.geometry.PointCloud()
        point_cloud.points = grasp_detect.o3d.utility.Vector3dVector([[0.0, 0.0, 0.0]])
        frame = {
            "origin": np.zeros(3), "x_axis": np.array([1.0, 0.0, 0.0]),
            "y_axis": np.array([0.0, 1.0, 0.0]), "z_axis": np.array([0.0, 0.0, 1.0]),
        }
        frames_result = (point_cloud, None, None, None, [frame], [], [], [], np.eye(4))
        with patch.object(grasp_detect, "frames_process", return_value=frames_result), patch.object(
            grasp_detect, "generate_cylinder_sections", return_value=(None, None, None, None)
        ):
            result = grasp_detect.grasp_detect("unused.ply", 1)

        self.assertEqual(len(result), 5)
        self.assertEqual(result[1], [])


if __name__ == "__main__":
    unittest.main()
