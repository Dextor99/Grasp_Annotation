import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class GraspScoringTests(unittest.TestCase):
    def setUp(self):
        self.object_data = SimpleNamespace(cloud_down=object())

    def test_uses_existing_scorer_and_ranks_force_closure_descending(self):
        from grasp_scoring import score_grasp_candidates

        candidates = [{"id": "low"}, {"id": "high"}]

        def fake_score(grasps, point_cloud, vis=False):
            self.assertIs(point_cloud, self.object_data.cloud_down)
            self.assertFalse(vis)
            grasps[0]["score_force_closure"] = 0.2
            grasps[0]["score_inner_points_ratio"] = 0.8
            grasps[1]["score_force_closure"] = 0.9
            grasps[1]["score_inner_points_ratio"] = 0.1
            return grasps

        with patch("grasp_scoring.compute_grasp_scores_simple", side_effect=fake_score) as scorer:
            scored = score_grasp_candidates(self.object_data, candidates)

        scorer.assert_called_once()
        self.assertEqual([grasp["id"] for grasp in scored], ["high", "low"])
        self.assertEqual([grasp["score_total"] for grasp in scored], [0.9, 0.2])
        self.assertIn("score_inner_points_ratio", scored[0])

    def test_falls_back_to_inner_ratio_and_always_returns_finite_total(self):
        from grasp_scoring import score_grasp_candidates

        candidates = [{"id": "fallback"}, {"id": "invalid"}]

        def fake_score(grasps, point_cloud, vis=False):
            grasps[0]["score_force_closure"] = np.nan
            grasps[0]["score_inner_points_ratio"] = 0.35
            grasps[1]["score_force_closure"] = None
            grasps[1]["score_inner_points_ratio"] = np.inf
            return grasps

        with patch("grasp_scoring.compute_grasp_scores_simple", side_effect=fake_score):
            scored = score_grasp_candidates(self.object_data, candidates)

        by_id = {grasp["id"]: grasp for grasp in scored}
        self.assertEqual(by_id["fallback"]["score_total"], 0.35)
        self.assertEqual(by_id["invalid"]["score_total"], 0.0)
        self.assertTrue(all(np.isfinite(grasp["score_total"]) for grasp in scored))

    def test_does_not_mutate_raw_candidate_records(self):
        from grasp_scoring import score_grasp_candidates

        candidate = {"id": "raw"}

        def fake_score(grasps, point_cloud, vis=False):
            grasps[0]["score_force_closure"] = 0.5
            return grasps

        with patch("grasp_scoring.compute_grasp_scores_simple", side_effect=fake_score):
            score_grasp_candidates(self.object_data, [candidate])

        self.assertNotIn("score_total", candidate)
        self.assertNotIn("score_force_closure", candidate)


class ScoredMultiViewAdapterTests(unittest.TestCase):
    def test_generates_raw_candidates_then_scores_them(self):
        from multi_view_grasp import generate_scored_multi_view_grasps

        object_data = SimpleNamespace(cloud_down=object())
        with patch("multi_view_grasp.generate_multi_view_grasps", return_value=[{"id": 1}]) as generate, patch(
            "multi_view_grasp.score_grasp_candidates", return_value=[{"id": 1, "score_total": 0.4}]
        ) as score:
            result = generate_scored_multi_view_grasps(
                "object.ply", object_data=object_data, num_views=1
            )

        generate.assert_called_once()
        self.assertIs(generate.call_args.kwargs["object_data"], object_data)
        score.assert_called_once_with(object_data, [{"id": 1}])
        self.assertEqual(result[0]["score_total"], 0.4)


if __name__ == "__main__":
    unittest.main()
