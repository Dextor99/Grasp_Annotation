import unittest
from dataclasses import FrozenInstanceError


class GraspGenerationConfigTests(unittest.TestCase):
    def test_freezes_requested_method_defaults(self):
        from grasp_config import GraspGenerationConfig

        config = GraspGenerationConfig()

        self.assertEqual(config.mode, "cone")
        self.assertEqual(config.cone_angle_deg, 15.0)
        self.assertEqual(config.num_approach_azimuth, 4)
        self.assertEqual(config.num_approach_directions, 5)
        self.assertEqual(config.normal_knn, 30)
        self.assertEqual(config.depth_samples, 16)
        self.assertEqual(config.translation_merge_mm, 5.0)
        self.assertEqual(config.rotation_merge_deg, 10.0)
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


if __name__ == "__main__":
    unittest.main()
