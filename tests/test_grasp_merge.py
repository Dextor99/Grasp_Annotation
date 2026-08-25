import unittest

import numpy as np

from grasp_merge import merge_duplicate_grasps, rotation_angle_degrees


class GraspMergeTests(unittest.TestCase):
    def test_keeps_higher_scored_duplicate(self):
        lower = {"T_gripper_object": np.eye(4), "score_total": 0.2}
        higher = {"T_gripper_object": np.eye(4), "score_total": 0.8}

        self.assertEqual(
            merge_duplicate_grasps([lower, higher], 5.0, 10.0), [higher]
        )

    def test_keeps_poses_outside_translation_or_rotation_threshold(self):
        translated = np.eye(4)
        translated[0, 3] = 6.0
        rotated = np.eye(4)
        theta = np.deg2rad(11.0)
        rotated[:3, :3] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]

        grasps = [
            {"T_gripper_object": np.eye(4), "score_total": 3.0},
            {"T_gripper_object": translated, "score_total": 2.0},
            {"T_gripper_object": rotated, "score_total": 1.0},
        ]

        self.assertEqual(len(merge_duplicate_grasps(grasps)), 3)

    def test_drops_nonfinite_or_invalid_homogeneous_poses(self):
        nonfinite = np.eye(4)
        nonfinite[0, 0] = np.nan
        invalid_rotation = np.eye(4)
        invalid_rotation[:3, :3] = 0.0

        result = merge_duplicate_grasps([
            {"T_gripper_object": nonfinite, "score_total": 4.0},
            {"T_gripper_object": invalid_rotation, "score_total": 3.0},
            {"T_gripper_object": np.eye(3), "score_total": 2.0},
            {"T_gripper_object": np.eye(4), "score_total": 1.0},
        ])

        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0]["T_gripper_object"], np.eye(4))

    def test_rotation_angle_clips_numerical_roundoff(self):
        rotation = np.eye(3)
        rotation[0, 0] = 1.0 + 1e-12

        self.assertEqual(rotation_angle_degrees(np.eye(3), rotation), 0.0)


if __name__ == "__main__":
    unittest.main()
