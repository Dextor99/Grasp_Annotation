import unittest

import numpy as np


def _cube_points(extent_x=0.04, extent_y=0.04, extent_z=0.04, step=0.01):
    values = np.arange(-0.5, 0.5001, step / max(extent_x, extent_y, extent_z))
    return np.array(
        [[x * extent_x, y * extent_y, z * extent_z] for x in values for y in values for z in values],
        dtype=np.float32,
    )


class GraspNetGeometryTests(unittest.TestCase):
    def test_adaptive_width_is_bounded_and_positive(self):
        from baselines.graspnet_annotation.gripper_geometry import estimate_opening_m

        points = _cube_points()
        width = estimate_opening_m(points, np.zeros(3), np.eye(3), max_width_m=0.12)
        self.assertGreater(width, 0.0)
        self.assertLessEqual(width, 0.12)

    def test_normal_grasp_has_nonempty_inner_region_without_finger_collision(self):
        from baselines.graspnet_annotation.gripper_geometry import evaluate_gripper_geometry

        result = evaluate_gripper_geometry(
            _cube_points(), np.zeros(3), np.eye(3), opening_m=0.06, depth_m=0.02,
            height_m=0.02, depth_base_m=0.02, finger_width_m=0.01, empty_thresh=10,
        )
        self.assertFalse(result.empty)
        self.assertFalse(result.collision)
        self.assertGreaterEqual(result.inner_count, 10)

    def test_empty_grasp_is_rejected_without_being_a_collision(self):
        from baselines.graspnet_annotation.gripper_geometry import evaluate_gripper_geometry

        points = _cube_points() + np.array([0.0, 0.2, 0.0], dtype=np.float32)
        result = evaluate_gripper_geometry(
            points, np.zeros(3), np.eye(3), opening_m=0.06, depth_m=0.02,
            height_m=0.02, depth_base_m=0.02, finger_width_m=0.01, empty_thresh=10,
        )
        self.assertTrue(result.empty)
        self.assertFalse(result.collision)

    def test_penetrating_finger_is_collision(self):
        from baselines.graspnet_annotation.gripper_geometry import evaluate_gripper_geometry

        points = np.vstack((_cube_points(), np.array([[0.0, 0.035, 0.0]], dtype=np.float32)))
        result = evaluate_gripper_geometry(
            points, np.zeros(3), np.eye(3), opening_m=0.06, depth_m=0.02,
            height_m=0.02, depth_base_m=0.02, finger_width_m=0.01, empty_thresh=10,
        )
        self.assertFalse(result.empty)
        self.assertTrue(result.collision)


if __name__ == "__main__":
    unittest.main()
