import unittest
import json
from dataclasses import FrozenInstanceError

import numpy as np


class GraspGenerationConfigTests(unittest.TestCase):
    def test_freezes_requested_method_defaults(self):
        from grasp_config import GraspGenerationConfig, V4_HIGH_QUALITY_THRESHOLD

        config = GraspGenerationConfig()

        self.assertEqual(config.mode, "cone")
        self.assertEqual(config.cone_angle_deg, 15.0)
        self.assertEqual(config.num_approach_azimuth, 4)
        self.assertEqual(config.num_approach_directions, 5)
        self.assertEqual(config.normal_knn, 30)
        self.assertEqual(config.depth_samples, 16)
        self.assertEqual(config.translation_merge_mm, 5.0)
        self.assertEqual(config.rotation_merge_deg, 10.0)
        self.assertEqual(config.closure_margin_mm, 2.0)
        self.assertEqual(V4_HIGH_QUALITY_THRESHOLD, 0.13)
        self.assertTrue(config.deterministic)
        self.assertEqual(config.random_seed, 0)
        with self.assertRaises(FrozenInstanceError):
            config.mode = "normal"

    def test_serializes_and_validates_runtime_parameters(self):
        from grasp_config import GraspGenerationConfig

        config = GraspGenerationConfig(num_views=3, anchors_per_view=2)
        self.assertEqual(config.to_dict()["num_views"], 3)
        with self.assertRaises(ValueError):
            GraspGenerationConfig(mode="invalid")
        with self.assertRaises(ValueError):
            GraspGenerationConfig(num_views=0)
        with self.assertRaises(ValueError):
            GraspGenerationConfig(rotation_max_deg=179.5)
        with self.assertRaises(ValueError):
            GraspGenerationConfig(num_approach_azimuth=True)
        with self.assertRaises(ValueError):
            GraspGenerationConfig(closure_margin_mm=0.0)

        numpy_config = GraspGenerationConfig(cone_angle_deg=np.float32(15.0))
        json.dumps(numpy_config.to_dict(), allow_nan=False)
        self.assertIs(type(numpy_config.to_dict()["cone_angle_deg"]), float)

    def test_v4_protocol_metadata_is_frozen_and_json_safe(self):
        import grasp_config

        metadata = {
            "threshold": grasp_config.V4_HIGH_QUALITY_THRESHOLD,
            "source": grasp_config.V4_THRESHOLD_SOURCE,
            "samples": grasp_config.V4_CALIBRATION_SAMPLES,
            "good": grasp_config.V4_CALIBRATION_GOOD,
            "bad": grasp_config.V4_CALIBRATION_BAD,
            "uncertain": grasp_config.V4_CALIBRATION_UNCERTAIN,
        }
        self.assertEqual(metadata["threshold"], 0.13)
        self.assertEqual(metadata["source"], "manual_calibration")
        self.assertEqual(metadata["samples"], 60)
        self.assertEqual(metadata["good"] + metadata["bad"] + metadata["uncertain"], 60)
        json.dumps(metadata, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
