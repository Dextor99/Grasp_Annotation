import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from grasp_config import GraspGenerationConfig


class GraspPipelineTests(unittest.TestCase):
    def test_runs_fixed_stage_order_and_separates_raw_from_unique(self):
        from grasp_pipeline import run_grasp_annotation

        events = []
        object_data = SimpleNamespace(
            scale=1.0,
            points=np.zeros((4, 3)),
            cloud_down=object(),
            T_object_world=np.eye(4),
        )
        raw = [{"id": 1}, {"id": 2}]
        scored = [{"id": 1, "score_total": 0.8}, {"id": 2, "score_total": 0.7}]
        unique = [scored[0]]

        with patch(
            "grasp_pipeline.configure_determinism",
            side_effect=lambda enabled, seed: events.append("determinism"),
        ), patch(
            "grasp_pipeline.prepare_object",
            side_effect=lambda path: events.append("prepare") or object_data,
        ), patch(
            "grasp_pipeline.generate_multi_view_grasps",
            side_effect=lambda *args, **kwargs: events.append("generate") or raw,
        ) as generate, patch(
            "grasp_pipeline.score_grasp_candidates",
            side_effect=lambda *args, **kwargs: events.append("score") or scored,
        ), patch(
            "grasp_pipeline.refine_grasp_closures",
            side_effect=lambda *args, **kwargs: events.append("refine") or scored,
        ) as refine, patch(
            "grasp_pipeline.validate_refined_grasp_closures",
            side_effect=lambda *args, **kwargs: events.append("validate") or scored,
        ) as validate, patch(
            "grasp_pipeline.merge_grasp_candidates",
            side_effect=lambda *args, **kwargs: events.append("merge") or unique,
        ) as merge, patch(
            "grasp_pipeline.normalize_grasp_record",
            side_effect=lambda grasp: events.append("normalize") or {"id": grasp["id"]},
        ):
            config = GraspGenerationConfig(num_views=1, anchors_per_view=1)
            result = run_grasp_annotation("model/object.ply", config=config)

        self.assertEqual(
            events[:7],
            ["determinism", "prepare", "generate", "score", "refine", "validate", "merge"],
        )
        self.assertEqual(result.raw_grasps, [{"id": 1}, {"id": 2}])
        self.assertEqual(result.unique_grasps, [{"id": 1}])
        self.assertEqual(result.meta["raw_grasp_count"], 2)
        self.assertEqual(result.meta["unique_grasp_count"], 1)
        self.assertEqual(
            result.meta["candidate_counts"],
            {
                "generated_candidate_count": 2,
                "scored_candidate_count": 2,
                "refinement_input_count": 2,
                "closure_geometry_rejected": 0,
                "closure_pose_collision_rejected": 0,
                "closure_valid_count": 2,
                "unique_grasp_count": 1,
            },
        )
        self.assertEqual(result.meta["merge_reduction_ratio"], 0.5)
        self.assertEqual(result.meta["units"], "mm")
        self.assertEqual(result.meta["config"]["mode"], "cone")
        self.assertIs(generate.call_args.kwargs["object_data"], object_data)
        self.assertIs(generate.call_args.kwargs["config"], config)
        self.assertEqual(merge.call_args.kwargs["translation_threshold_mm"], 5.0)
        self.assertEqual(merge.call_args.kwargs["rotation_threshold_deg"], 10.0)
        self.assertEqual(refine.call_args.kwargs["margin_mm"], 2.0)
        self.assertIs(validate.call_args.kwargs["point_cloud"], object_data.cloud_down)
        self.assertEqual(validate.call_args.kwargs["threshold_mm"], 3.0)


if __name__ == "__main__":
    unittest.main()
