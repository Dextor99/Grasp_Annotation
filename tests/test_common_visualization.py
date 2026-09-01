import unittest

import numpy as np


class CommonVisualizationTests(unittest.TestCase):
    def test_cli_converts_path_to_string_for_open3d_loader(self):
        from pathlib import Path
        from scripts.common_eval.visualize_common_topk import object_loader_argument

        self.assertEqual(object_loader_argument(Path("model/2.ply")), str(Path("model/2.ply")))

    def test_official_candidate_conversion_preserves_center_and_maps_axes(self):
        from scripts.common_eval.visualize_common_topk import official_candidate_to_record

        record = official_candidate_to_record(
            point_m=np.array([0.1, 0.2, 0.3]),
            rotation_official=np.eye(3),
            depth_m=0.02,
            width_m=0.04,
            object_world=np.eye(4),
        )
        np.testing.assert_allclose(record["translation"], [120.0, 200.0, 300.0])
        self.assertAlmostEqual(record["depth_mm"], 20.0)
        self.assertAlmostEqual(record["grasp_width_mm"], 40.0)
        self.assertTrue(np.isfinite(np.asarray(record["rotation_matrix"])).all())


if __name__ == "__main__":
    unittest.main()
