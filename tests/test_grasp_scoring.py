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

    def test_configures_determinism_before_preparing_object(self):
        from grasp_config import GraspGenerationConfig
        from multi_view_grasp import generate_scored_multi_view_grasps

        events = []
        object_data = SimpleNamespace(cloud_down=object())
        config = GraspGenerationConfig(random_seed=41)
        with patch(
            "multi_view_grasp.configure_determinism",
            side_effect=lambda enabled, seed: events.append(("seed", enabled, seed)),
        ), patch(
            "multi_view_grasp.prepare_object",
            side_effect=lambda path: events.append(("prepare", path)) or object_data,
        ), patch("multi_view_grasp.generate_multi_view_grasps", return_value=[]), patch(
            "multi_view_grasp.score_grasp_candidates", return_value=[]
        ):
            generate_scored_multi_view_grasps("object.ply", config=config)

        self.assertEqual(events[0], ("seed", True, 41))
        self.assertEqual(events[1], ("prepare", "object.ply"))


if __name__ == "__main__":
    unittest.main()
