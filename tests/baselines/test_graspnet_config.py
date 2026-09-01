import unittest


class DenseAnnotationConfigTests(unittest.TestCase):
    def test_full_defaults_match_graspnet_label_topology(self):
        from baselines.graspnet_annotation.config import DenseAnnotationConfig

        config = DenseAnnotationConfig.full()
        self.assertEqual((config.num_views, config.num_angles, config.depths_m), (300, 12, (0.01, 0.02, 0.03, 0.04)))
        self.assertEqual(config.candidates_per_point, 14_400)
        self.assertEqual(config.input_unit, "m")

    def test_debug_override_has_exact_cartesian_candidate_count(self):
        from baselines.graspnet_annotation.config import DenseAnnotationConfig

        config = DenseAnnotationConfig.full(num_views=3, num_angles=3, depths_m=(0.01, 0.02))
        self.assertEqual(config.candidates_per_point, 18)

    def test_rejects_unknown_unit_and_nonpositive_gripper_values(self):
        from baselines.graspnet_annotation.config import DenseAnnotationConfig

        with self.assertRaisesRegex(ValueError, "input_unit"):
            DenseAnnotationConfig.full(input_unit="unit_sphere")
        with self.assertRaisesRegex(ValueError, "max_width_m"):
            DenseAnnotationConfig.full(max_width_m=0.0)


if __name__ == "__main__":
    unittest.main()
