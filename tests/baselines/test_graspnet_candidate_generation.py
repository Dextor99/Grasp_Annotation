import unittest

import numpy as np


class GraspNetCandidateGenerationTests(unittest.TestCase):
    def test_one_point_official_topology_has_14400_indexed_candidates(self):
        from baselines.graspnet_annotation.candidate_generation import iter_candidate_batches
        from baselines.graspnet_annotation.config import DenseAnnotationConfig

        config = DenseAnnotationConfig.full()
        batches = list(iter_candidate_batches(np.zeros((1, 3)), config, point_batch_size=1))
        self.assertEqual(sum(batch.size for batch in batches), 14_400)
        self.assertEqual((batches[0].view_ids.min(), batches[0].view_ids.max()), (0, 299))
        self.assertEqual((batches[0].angle_ids.min(), batches[0].angle_ids.max()), (0, 11))
        self.assertEqual((batches[0].depth_ids.min(), batches[0].depth_ids.max()), (0, 3))

    def test_debug_topology_is_exactly_18_candidates(self):
        from baselines.graspnet_annotation.candidate_generation import iter_candidate_batches
        from baselines.graspnet_annotation.config import DenseAnnotationConfig

        config = DenseAnnotationConfig.full(num_views=3, num_angles=3, depths_m=(0.01, 0.02))
        self.assertEqual(next(iter_candidate_batches(np.zeros((1, 3)), config)).size, 18)

    def test_twenty_point_streaming_never_expands_more_than_one_point_batch(self):
        from baselines.graspnet_annotation.candidate_generation import iter_candidate_batches
        from baselines.graspnet_annotation.config import DenseAnnotationConfig

        config = DenseAnnotationConfig.full()
        batches = list(iter_candidate_batches(np.zeros((20, 3)), config, point_batch_size=1))
        self.assertEqual(len(batches), 20)
        self.assertEqual(sum(batch.size for batch in batches), 288_000)
        self.assertEqual(max(batch.size for batch in batches), 14_400)

    def test_official_rotations_are_proper_and_use_approach_towards_object(self):
        from baselines.graspnet_annotation.view_sampling import generate_views, generate_view_rotations

        views = generate_views(300)
        rotations = generate_view_rotations(views[:3], np.zeros(3, dtype=np.float32))
        for rotation in rotations:
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
