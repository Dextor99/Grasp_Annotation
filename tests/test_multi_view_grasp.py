import unittest
from unittest.mock import patch

import numpy as np

from multi_view_grasp import generate_multi_view_grasps


class MultiViewGraspTests(unittest.TestCase):
    def test_calls_detector_once_per_requested_view_and_attaches_provenance(self):
        calls = []

        def detector(points, normals, view, metadata):
            calls.append((points.copy(), normals.copy(), view.copy(), metadata.copy()))
            return [{"T_gripper_object": np.eye(4), "opening": 50.0}]

        with patch("multi_view_grasp.filter_front_facing_surface", side_effect=lambda points, normals, view: (points, normals, np.ones(len(points), dtype=bool))):
            result = generate_multi_view_grasps(
                "ignored", 3,
                loader=lambda _: (np.zeros((3, 3)), np.tile([0.0, 0.0, 1.0], (3, 1))),
                detector=detector,
                scorer=lambda grasps, _: grasps,
                deduplicate=False,
            )

        self.assertEqual([call[3]["view_id"] for call in calls], [0, 1, 2])
        self.assertEqual(result.view_candidate_counts, {0: 1, 1: 1, 2: 1})
        self.assertEqual([grasp["view_id"] for grasp in result.grasps], [0, 1, 2])
        for grasp, call in zip(result.grasps, calls):
            np.testing.assert_allclose(grasp["view_direction"], call[2])
            self.assertAlmostEqual(np.linalg.norm(grasp["view_direction"]), 1.0)

    def test_records_empty_and_candidate_free_views_as_skipped(self):
        points = np.array([[0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0]])
        visible = [
            (np.empty((0, 3)), np.empty((0, 3)), np.array([False])),
            (points, normals, np.array([True])),
        ]

        with patch("multi_view_grasp.filter_front_facing_surface", side_effect=visible) as filter_surface:
            result = generate_multi_view_grasps(
                "ignored", 2,
                loader=lambda _: (points, normals),
                detector=lambda *_: [],
                scorer=lambda grasps, _: grasps,
            )

        self.assertEqual(filter_surface.call_count, 2)
        self.assertEqual(result.grasps, [])
        self.assertEqual(result.view_candidate_counts, {0: 0, 1: 0})
        self.assertEqual(result.skipped_views, [
            {"view_id": 0, "reason": "no_front_facing_points"},
            {"view_id": 1, "reason": "no_candidates"},
        ])

    def test_score_preference_and_deduplication_happen_after_all_views(self):
        lower = {"T_gripper_object": np.eye(4), "opening": 20.0, "score_inner_points_ratio": 0.7}
        higher = {"T_gripper_object": np.eye(4), "opening": 30.0, "score_force_closure": 0.9}
        batches = [[lower], [higher]]

        def detector(*_):
            return batches.pop(0)

        with patch("multi_view_grasp.filter_front_facing_surface", side_effect=lambda points, normals, view: (points, normals, np.ones(len(points), dtype=bool))):
            result = generate_multi_view_grasps(
                "ignored", 2,
                loader=lambda _: (np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]])),
                detector=detector,
                scorer=lambda grasps, _: grasps,
            )

        self.assertEqual(len(result.grasps), 1)
        self.assertEqual(result.grasps[0]["opening"], 30.0)
        self.assertEqual(result.grasps[0]["score_total"], 0.9)

    def test_score_total_uses_negative_infinity_without_finite_scores(self):
        candidate = {"T_gripper_object": np.eye(4), "score_force_closure": np.nan, "score_inner_points_ratio": None}
        with patch("multi_view_grasp.filter_front_facing_surface", return_value=(np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), np.array([True]))):
            result = generate_multi_view_grasps(
                "ignored", 1,
                loader=lambda _: (np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]])),
                detector=lambda *_: [candidate],
                scorer=lambda grasps, _: grasps,
                deduplicate=False,
            )

        self.assertEqual(result.grasps[0]["score_total"], float("-inf"))


if __name__ == "__main__":
    unittest.main()
