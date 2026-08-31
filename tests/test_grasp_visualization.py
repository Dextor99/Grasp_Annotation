import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _record(score=0.5, translation=(0.0, 0.0, 0.0), opening=20.0):
    return {
        "translation": list(translation),
        "rotation_matrix": np.eye(3).tolist(),
        "opening_mm": opening,
        "depth_mm": 10.0,
        "score_total": score,
        "view_id": 0,
        "anchor_id": 0,
        "approach_id": 0,
        "anchor_point": [0.0, 0.0, 0.0],
        "anchor_normal": [0.0, 0.0, 1.0],
        "approach_direction": [0.0, 0.0, 1.0],
    }


class GraspVisualizationTests(unittest.TestCase):
    def test_record_to_transform(self):
        from grasp_visualization import record_to_transform

        record = _record(translation=(1.0, 2.0, 3.0))
        transform = record_to_transform(record)

        expected = np.eye(4)
        expected[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_array_equal(transform, expected)

    def test_select_grasps_sorts_by_score(self):
        from grasp_visualization import select_grasps

        records = [_record(0.2), _record(0.9), _record(0.5)]
        selected = select_grasps(records, topk=2)

        self.assertEqual([r["score_total"] for r in selected], [0.9, 0.5])

    def test_line_gripper_respects_opening(self):
        from grasp_visualization import make_gripper_lineset

        geometry = make_gripper_lineset(_record(opening=20.0), finger_length=10.0, finger_thickness=4.0)
        points = np.asarray(geometry.points)

        self.assertAlmostEqual(points[0, 1], -12.0)
        self.assertAlmostEqual(points[1, 1], 12.0)

    def test_line_gripper_respects_pose(self):
        from grasp_visualization import make_gripper_lineset

        record = _record(translation=(5.0, 6.0, 7.0))
        geometry = make_gripper_lineset(record, finger_length=10.0)
        points = np.asarray(geometry.points)

        np.testing.assert_allclose(points[0, :], [5.0, -6.5, 7.0])
        np.testing.assert_allclose(points[2, :], [5.0, -6.5, -3.0])

    def test_load_grasp_records_and_meta(self):
        from grasp_visualization import load_grasp_records, load_meta

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "grasps.json").write_text(json.dumps([_record()]), encoding="utf-8")
            (root / "meta.json").write_text(json.dumps({"units": "mm"}), encoding="utf-8")
            self.assertEqual(len(load_grasp_records(root)), 1)
            self.assertEqual(load_meta(root)["units"], "mm")


if __name__ == "__main__":
    unittest.main()
