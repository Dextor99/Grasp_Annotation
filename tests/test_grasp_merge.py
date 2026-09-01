import unittest

import numpy as np


def _pose(translation, rotation=None):
    transform = np.eye(4)
    transform[:3, :3] = np.eye(3) if rotation is None else rotation
    transform[:3, 3] = translation
    return transform


def _grasp(score, translation=(0.0, 0.0, 0.0), rotation=None, **metadata):
    return {
        "T_gripper_object": _pose(translation, rotation),
        "score_total": score,
        **metadata,
    }


class GraspMergeTests(unittest.TestCase):
    def test_merges_nearby_pose_and_retains_best_score(self):
        from grasp_merge import merge_grasp_candidates

        low = _grasp(0.2, translation=(0.0, 0.0, 0.0), view_id=1, anchor_id=2, approach_id=3)
        high = _grasp(0.9, translation=(4.9, 0.0, 0.0), view_id=4, anchor_id=5, approach_id=6)

        merged = merge_grasp_candidates([low, high])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["score_total"], 0.9)
        self.assertEqual(merged[0]["source_view_ids"], [1, 4])
        self.assertEqual(merged[0]["source_anchor_ids"], [2, 5])
        self.assertEqual(merged[0]["source_approach_ids"], [3, 6])
        self.assertEqual(
            merged[0]["source_ids"],
            [
                {"view_id": 1, "anchor_id": 2, "approach_id": 3},
                {"view_id": 4, "anchor_id": 5, "approach_id": 6},
            ],
        )

    def test_parallel_jaw_half_turn_is_physically_equivalent(self):
        from grasp_merge import merge_grasp_candidates

        half_turn_local_z = np.diag([-1.0, -1.0, 1.0])
        grasps = [
            _grasp(0.7, rotation=np.eye(3), view_id=0, anchor_id=0, approach_id=0),
            _grasp(0.6, rotation=half_turn_local_z, view_id=1, anchor_id=1, approach_id=1),
        ]

        self.assertEqual(len(merge_grasp_candidates(grasps)), 1)

    def test_half_turn_is_applied_about_local_not_world_z(self):
        from grasp_merge import merge_grasp_candidates

        angle = np.deg2rad(37.0)
        base = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, np.cos(angle), -np.sin(angle)],
             [0.0, np.sin(angle), np.cos(angle)]]
        )
        local_half_turn = np.diag([-1.0, -1.0, 1.0])
        grasps = [_grasp(0.8, rotation=base), _grasp(0.7, rotation=base @ local_half_turn)]

        self.assertEqual(len(merge_grasp_candidates(grasps)), 1)

    def test_half_turn_about_non_symmetry_axis_remains_distinct(self):
        from grasp_merge import merge_grasp_candidates

        local_x_half_turn = np.diag([1.0, -1.0, -1.0])
        grasps = [_grasp(0.8, rotation=np.eye(3)), _grasp(0.7, rotation=local_x_half_turn)]

        self.assertEqual(len(merge_grasp_candidates(grasps)), 2)

    def test_keeps_candidates_at_or_beyond_strict_thresholds(self):
        from grasp_merge import merge_grasp_candidates

        angle = np.deg2rad(10.0)
        rotation_10_deg = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        translation_boundary = [
            _grasp(0.9, translation=(0.0, 0.0, 0.0)),
            _grasp(0.8, translation=(5.0, 0.0, 0.0)),
        ]
        rotation_boundary = [
            _grasp(0.9, rotation=np.eye(3)),
            _grasp(0.8, rotation=rotation_10_deg),
        ]

        self.assertEqual(len(merge_grasp_candidates(translation_boundary)), 2)
        self.assertEqual(len(merge_grasp_candidates(rotation_boundary)), 2)

    def test_keeps_distinct_poses_and_orders_representatives_by_score(self):
        from grasp_merge import merge_grasp_candidates

        grasps = [
            _grasp(0.2, translation=(20.0, 0.0, 0.0)),
            _grasp(0.8, translation=(0.0, 0.0, 0.0)),
            _grasp(0.5, translation=(40.0, 0.0, 0.0)),
        ]

        merged = merge_grasp_candidates(grasps)

        self.assertEqual([grasp["score_total"] for grasp in merged], [0.8, 0.5, 0.2])

    def test_greedy_merge_does_not_apply_transitive_closure(self):
        from grasp_merge import merge_grasp_candidates

        grasps = [
            _grasp(0.9, translation=(0.0, 0.0, 0.0)),
            _grasp(0.8, translation=(4.0, 0.0, 0.0)),
            _grasp(0.7, translation=(8.0, 0.0, 0.0)),
        ]

        self.assertEqual(len(merge_grasp_candidates(grasps)), 2)

    def test_unions_existing_provenance_without_mutating_inputs(self):
        from grasp_merge import merge_grasp_candidates

        first = _grasp(
            0.8,
            view_id=2,
            anchor_id=3,
            approach_id=4,
            source_view_ids=[0, 2],
            source_anchor_ids=[1, 3],
            source_approach_ids=[2, 4],
        )
        second = _grasp(0.7, translation=(1.0, 0.0, 0.0), view_id=5, anchor_id=6, approach_id=7)

        merged = merge_grasp_candidates([first, second])

        self.assertEqual(merged[0]["source_view_ids"], [0, 2, 5])
        self.assertEqual(merged[0]["source_anchor_ids"], [1, 3, 6])
        self.assertEqual(merged[0]["source_approach_ids"], [2, 4, 7])
        self.assertEqual(first["source_view_ids"], [0, 2])

    def test_rejects_non_finite_scores_and_invalid_poses(self):
        from grasp_merge import merge_grasp_candidates

        with self.assertRaises(ValueError):
            merge_grasp_candidates([_grasp(np.nan)])
        with self.assertRaises(ValueError):
            merge_grasp_candidates([{"T_gripper_object": np.eye(3), "score_total": 0.5}])

    def test_can_select_representative_with_explicit_v4_score(self):
        from grasp_merge import merge_grasp_candidates

        v3_high = _grasp(0.99, translation=(0.0, 0.0, 0.0), score_total_v4=0.2)
        v4_high = _grasp(0.80, translation=(1.0, 0.0, 0.0), score_total_v4=0.9)

        merged = merge_grasp_candidates([v3_high, v4_high], score_key="score_total_v4")

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["score_total"], 0.80)
        self.assertEqual(merged[0]["score_total_v4"], 0.9)


if __name__ == "__main__":
    unittest.main()
