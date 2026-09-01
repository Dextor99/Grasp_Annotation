import unittest
import json

import numpy as np


class GraspSchemaTests(unittest.TestCase):
    def _grasp(self):
        return {
            "T_gripper_object": np.eye(4),
            "opening": 45.0,
            "depth": 20.0,
            "score_total": 0.8,
            "score_force_closure": 0.8,
            "view_id": 1,
            "anchor_id": 2,
            "approach_id": 3,
            "view_direction": [0.0, 0.0, 1.0],
            "anchor_point": [1.0, 2.0, 3.0],
            "anchor_normal": [0.0, 0.0, 1.0],
            "approach_direction": [0.0, 0.0, 1.0],
            "source_view_ids": [1, 4],
            "source_anchor_ids": [2, 5],
            "source_approach_ids": [3, 6],
            "source_ids": [
                {"view_id": 1, "anchor_id": 2, "approach_id": 3},
                {"view_id": 4, "anchor_id": 5, "approach_id": 6},
            ],
        }

    def test_normalizes_required_fields_and_xyzw_quaternion(self):
        from grasp_schema import normalize_grasp_record

        record = normalize_grasp_record(self._grasp())

        required = {
            "translation", "rotation_matrix", "quaternion_xyzw", "opening_mm",
            "depth_mm", "score_total", "view_id", "anchor_id", "approach_id",
            "view_direction", "anchor_point", "anchor_normal", "approach_direction",
            "source_view_ids", "source_anchor_ids", "source_approach_ids", "source_ids",
        }
        self.assertTrue(required.issubset(record))
        self.assertEqual(record["translation"], [0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(record["quaternion_xyzw"], [0.0, 0.0, 0.0, 1.0]))
        self.assertEqual(record["opening_mm"], 45.0)
        self.assertEqual(record["search_opening_mm"], 45.0)
        self.assertEqual(record["grasp_width_mm"], 45.0)
        self.assertEqual(record["support_span_mm"], 0.0)
        self.assertEqual(record["requested_margin_mm"], 0.0)
        self.assertEqual(record["effective_margin_mm"], 0.0)
        self.assertFalse(record["closure_refined"])
        self.assertTrue(record["closure_geometry_valid"])
        self.assertTrue(record["closure_pose_valid"])
        self.assertIsNone(record["score_y0_diff_refined"])
        self.assertEqual(record["depth_mm"], 20.0)
        self.assertEqual(record["score_force_closure"], 0.8)

    def test_rejects_non_finite_values_and_missing_metadata(self):
        from grasp_schema import normalize_grasp_record

        grasp = self._grasp()
        grasp["score_total"] = np.nan
        with self.assertRaises(ValueError):
            normalize_grasp_record(grasp)

    def test_rejects_final_width_wider_than_search_opening(self):
        from grasp_schema import normalize_grasp_record

        grasp = self._grasp()
        grasp["search_opening_mm"] = 40.0
        grasp["grasp_width_mm"] = 45.0
        with self.assertRaises(ValueError):
            normalize_grasp_record(grasp)

    def test_exports_v4_total_as_final_score_and_preserves_v3(self):
        from grasp_schema import normalize_grasp_record

        grasp = self._grasp()
        grasp.update(
            {
                "score_total_v3": 0.91,
                "score_total_v4": 0.37,
                "score_v4_normal": 0.8,
                "score_v4_support": 0.2,
                "score_v4_stability": 0.9,
                "score_v4_normal_dispersion": 0.95,
            }
        )

        record = normalize_grasp_record(grasp)

        self.assertEqual(record["score_total"], 0.37)
        self.assertEqual(record["score_total_v3"], 0.91)
        self.assertEqual(record["score_total_v4"], 0.37)
        self.assertEqual(record["score_v4_support"], 0.2)

    def test_numpy_provenance_ids_are_converted_to_json_safe_ints(self):
        from grasp_schema import normalize_grasp_record

        grasp = self._grasp()
        grasp["view_id"] = np.int64(1)
        grasp["source_view_ids"] = [np.int64(1), np.int64(4)]
        grasp["source_ids"] = [
            {"view_id": np.int64(1), "anchor_id": np.int64(2), "approach_id": np.int64(3)}
        ]

        record = normalize_grasp_record(grasp)

        json.dumps(record, allow_nan=False)
        self.assertTrue(all(type(value) is int for value in record["source_view_ids"]))
        self.assertIs(type(record["source_ids"][0]["view_id"]), int)

        grasp = self._grasp()
        del grasp["anchor_normal"]
        with self.assertRaises(ValueError):
            normalize_grasp_record(grasp)


if __name__ == "__main__":
    unittest.main()
