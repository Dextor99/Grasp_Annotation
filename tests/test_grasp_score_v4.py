import unittest
from types import SimpleNamespace

import numpy as np


def _object_data(points, normals, center=None, radius=50.0):
    return SimpleNamespace(
        points=np.asarray(points, dtype=float),
        normals=np.asarray(normals, dtype=float),
        center=np.zeros(3) if center is None else np.asarray(center, dtype=float),
        radius=float(radius),
        T_object_world=np.eye(4),
    )


def _record(translation=(0.0, 0.0, 0.0), score_total=0.42):
    return {
        "translation": list(translation),
        "rotation_matrix": np.eye(3).tolist(),
        "opening_mm": 50.0,
        "grasp_width_mm": 40.0,
        "depth_mm": 20.0,
        "score_total": score_total,
    }


class GraspScoreV4Tests(unittest.TestCase):
    def _balanced_cloud(self, outlier=False, one_sided=False):
        points = []
        normals = []
        for side, normal_y in ((-20.0, -1.0), (20.0, 1.0)):
            if one_sided and side > 0:
                # Two collinear points cannot form a support area.
                grid = [(-2.0, -20.0), (2.0, -20.0)]
            else:
                grid = [(x, z) for x in (-2.0, 0.0, 2.0) for z in (-30.0, -20.0, -10.0)]
            for x, z in grid:
                points.append([x, side, z])
                normal = [0.0, normal_y, 0.0]
                if outlier and side < 0 and len(grid) == 9 and len(normals) == 0:
                    normal = [0.0, 1.0, 0.0]
                normals.append(normal)
        return _object_data(points, normals)

    def test_robust_contact_normals_ignore_single_outlier(self):
        from grasp_score_v4 import score_grasp_v4

        result = score_grasp_v4(
            _record(), self._balanced_cloud(outlier=True), voxel_size_mm=1.0
        )

        self.assertGreater(result["score_v4_normal"], 0.95)
        self.assertGreater(result["score_v4_normal_dispersion"], 0.8)
        self.assertEqual(result["contact_points_left"], 9)
        self.assertEqual(result["contact_points_right"], 9)

    def test_support_is_bilateral_geometric_mean(self):
        from grasp_score_v4 import score_grasp_v4

        balanced = score_grasp_v4(_record(), self._balanced_cloud(), voxel_size_mm=1.0)
        one_sided = score_grasp_v4(
            _record(), self._balanced_cloud(one_sided=True), voxel_size_mm=1.0
        )

        self.assertGreater(balanced["score_v4_support"], 0.0)
        self.assertEqual(one_sided["score_v4_support"], 0.0)
        self.assertLess(one_sided["score_total_v4"], balanced["score_total_v4"])

    def test_total_is_geometric_mean_and_v3_is_preserved(self):
        from grasp_score_v4 import score_grasp_v4

        result = score_grasp_v4(_record(score_total=0.73), self._balanced_cloud(), voxel_size_mm=1.0)
        expected = (
            result["score_v4_normal"]
            * result["score_v4_support"]
            * result["score_v4_stability"]
        ) ** (1.0 / 3.0)

        self.assertAlmostEqual(result["score_total_v3"], 0.73)
        self.assertAlmostEqual(result["score_total_v4"], expected)
        self.assertAlmostEqual(result["score_v4_stability"], 1.0)
        self.assertTrue(np.isfinite(result["score_total_v4"]))

    def test_offline_rescore_keeps_original_record_indices(self):
        from scripts.rescore_v4 import rescore_records

        records = [_record(translation=(40.0, 0.0, 0.0), score_total=0.99), _record(score_total=0.10)]
        rows, summary = rescore_records(records, self._balanced_cloud(), topk=1)

        self.assertEqual(summary["old_top1_record_index"], 0)
        self.assertEqual(summary["v4_top1_record_index"], 1)
        self.assertEqual({row["record_index"] for row in rows}, {0, 1})


if __name__ == "__main__":
    unittest.main()
