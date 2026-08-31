import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from multi_view_grasp import generate_multi_view_grasps
from grasp_config import GraspGenerationConfig


class MultiViewGraspTests(unittest.TestCase):
    def setUp(self):
        self.object_data = SimpleNamespace(
            points=np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], dtype=float),
            normals=np.tile([0, 0, 1], (4, 1)),
        )

    def test_cone_mode_runs_two_anchors_and_five_approaches(self):
        def fake_detect(*args, **kwargs):
            return [{"id": 1}]

        with patch("multi_view_grasp.grasp_detect_from_anchor_approach", side_effect=fake_detect) as detect:
            grasps = generate_multi_view_grasps(
                "object.ply",
                num_views=1,
                mode="cone",
                num_anchors_per_view=2,
                num_approach_azimuth=4,
                object_data=self.object_data,
            )

        self.assertEqual(detect.call_count, 10)
        self.assertEqual(len(grasps), 10)
        self.assertEqual({grasp["anchor_id"] for grasp in grasps}, {0, 1})
        self.assertEqual({grasp["approach_id"] for grasp in grasps}, {0, 1, 2, 3, 4})
        for grasp in grasps:
            for key in (
                "view_id", "view_direction", "anchor_point", "anchor_normal",
                "approach_direction", "approach_offset_deg",
            ):
                self.assertIn(key, grasp)

    def test_normal_mode_uses_one_approach_per_anchor(self):
        with patch("multi_view_grasp.grasp_detect_from_anchor_approach", return_value=[{"id": 1}]) as detect:
            grasps = generate_multi_view_grasps(
                "object.ply",
                num_views=1,
                mode="normal",
                num_anchors_per_view=2,
                object_data=self.object_data,
            )
        self.assertEqual(detect.call_count, 2)
        self.assertEqual(len(grasps), 2)
        self.assertEqual({grasp["approach_id"] for grasp in grasps}, {0})

    def test_global_mode_preserves_centroid_surface_path(self):
        with patch("multi_view_grasp.grasp_detect_from_surface", return_value=[{"id": 1}]) as detect:
            grasps = generate_multi_view_grasps(
                "object.ply",
                num_views=1,
                mode="global",
                object_data=self.object_data,
            )
        self.assertEqual(detect.call_count, 1)
        self.assertEqual(len(grasps), 1)
        self.assertEqual(grasps[0]["anchor_id"], -1)

    def test_prepares_object_only_once(self):
        with patch("multi_view_grasp.prepare_object", return_value=self.object_data) as prepare, patch(
            "multi_view_grasp.grasp_detect_from_surface", return_value=[]
        ):
            generate_multi_view_grasps("object.ply", num_views=3, mode="global")
        prepare.assert_called_once_with("object.ply")

    def test_central_config_drives_generation_and_reaches_detector(self):
        config = GraspGenerationConfig(num_views=1, anchors_per_view=1)
        with patch("multi_view_grasp.grasp_detect_from_anchor_approach", return_value=[]) as detect:
            generate_multi_view_grasps(
                "object.ply",
                object_data=self.object_data,
                config=config,
            )

        self.assertEqual(detect.call_count, 5)
        self.assertTrue(all(call.kwargs["config"] is config for call in detect.call_args_list))


if __name__ == "__main__":
    unittest.main()
