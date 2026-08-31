import unittest

import numpy as np


def _candidate(inner_points):
    return {
        "T_gripper_object": np.eye(4),
        "opening": 60.0,
        "inner_points_local": np.asarray(inner_points, dtype=float),
    }


class GraspRefinementTests(unittest.TestCase):
    def test_centers_pose_and_derives_final_grasp_width(self):
        from grasp_refinement import refine_grasp_closure

        refined = refine_grasp_closure(
            _candidate([[-1.0, -10.0, -20.0], [0.0, 20.0, -30.0]]),
            margin_mm=2.0,
        )

        self.assertEqual(refined["search_opening_mm"], 60.0)
        self.assertEqual(refined["grasp_width_mm"], 34.0)
        self.assertEqual(refined["closure_center_offset_mm"], 5.0)
        np.testing.assert_allclose(refined["T_gripper_object"][:3, 3], [0.0, 5.0, 0.0])
        self.assertTrue(refined["closure_refined"])

    def test_missing_inner_points_keep_search_opening(self):
        from grasp_refinement import refine_grasp_closure

        candidate = _candidate([])
        refined = refine_grasp_closure(candidate)

        self.assertEqual(refined["grasp_width_mm"], 60.0)
        self.assertEqual(refined["closure_center_offset_mm"], 0.0)
        self.assertFalse(refined["closure_refined"])

    def test_refinement_does_not_mutate_candidate(self):
        from grasp_refinement import refine_grasp_closure

        candidate = _candidate([[-1.0, -10.0, -20.0], [0.0, 20.0, -30.0]])
        original_pose = candidate["T_gripper_object"].copy()
        refine_grasp_closure(candidate)

        np.testing.assert_array_equal(candidate["T_gripper_object"], original_pose)
        self.assertNotIn("grasp_width_mm", candidate)


if __name__ == "__main__":
    unittest.main()
