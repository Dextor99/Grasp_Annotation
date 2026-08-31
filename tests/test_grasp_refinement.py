import unittest
from unittest.mock import patch

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
        np.testing.assert_allclose(refined["inner_points_local"][:, 1], [-15.0, 15.0])
        self.assertTrue(refined["closure_refined"])

    def test_caps_final_width_and_records_effective_margin(self):
        from grasp_refinement import refine_grasp_closure

        refined = refine_grasp_closure(
            {**_candidate([[-1.0, -29.5, -20.0], [0.0, 29.5, -30.0]]), "opening": 60.0},
            margin_mm=2.0,
        )

        self.assertEqual(refined["support_span_mm"], 59.0)
        self.assertEqual(refined["grasp_width_mm"], 60.0)
        self.assertEqual(refined["requested_margin_mm"], 2.0)
        self.assertEqual(refined["effective_margin_mm"], 0.5)
        self.assertTrue(refined["closure_geometry_valid"])

    def test_rejects_support_span_wider_than_search_opening(self):
        from grasp_refinement import refine_grasp_closure

        refined = refine_grasp_closure(
            {**_candidate([[-1.0, -31.0, -20.0], [0.0, 31.0, -30.0]]), "opening": 60.0},
            margin_mm=2.0,
        )

        self.assertFalse(refined["closure_geometry_valid"])
        self.assertEqual(refined["grasp_width_mm"], 60.0)
        self.assertEqual(refined["closure_center_offset_mm"], 0.0)

    def test_refined_diagnostic_y0_is_centered(self):
        from grasp_refinement import refine_grasp_closure

        refined = refine_grasp_closure(
            {**_candidate([[-1.0, -10.0, -20.0], [0.0, 20.0, -30.0]]), "score_y0_diff": 30.0},
            margin_mm=2.0,
        )

        self.assertEqual(refined["score_y0_diff_before_refinement"], 30.0)
        self.assertAlmostEqual(refined["score_y0_diff_refined"], 0.0)

    def test_post_refinement_validation_rebuilds_search_meshes(self):
        from grasp_refinement import validate_refined_grasp_closures

        candidate = {
            **_candidate([[-1.0, -10.0, -20.0], [0.0, 20.0, -30.0]]),
            "search_opening_mm": 60.0,
            "grasp_width_mm": 34.0,
            "closure_geometry_valid": True,
        }
        point_cloud = object()
        object_to_world = np.eye(4)
        fake_meshes = [object(), object(), object()]

        with patch("grasp_refinement.CollisionIndex.from_point_cloud", return_value=object()), patch(
            "grasp_refinement.create_gripper_model",
            return_value={"meshes": fake_meshes},
        ) as create_model, patch(
            "grasp_refinement.check_collision", return_value=False
        ) as check_collision:
            validated = validate_refined_grasp_closures(
                [candidate], point_cloud, object_to_world, threshold_mm=3.0
            )

        self.assertEqual(len(validated), 1)
        self.assertTrue(validated[0]["closure_pose_valid"])
        self.assertIs(validated[0]["meshes"], fake_meshes)
        self.assertEqual(create_model.call_args.kwargs["opening"], 60.0)
        self.assertIs(check_collision.call_args.args[1], point_cloud)

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
