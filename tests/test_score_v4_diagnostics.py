import unittest
from types import SimpleNamespace

import numpy as np


class ScoreV4DiagnosticsTests(unittest.TestCase):
    def test_script_resolves_project_imports(self):
        from pathlib import Path
        import scripts.score_v4_diagnostics as diagnostics

        self.assertEqual(diagnostics.PROJECT_ROOT, Path(diagnostics.__file__).resolve().parents[1])

    def test_robust_metrics_use_contact_bands_and_center_distance(self):
        from scripts.score_v4_diagnostics import compute_diagnostic_metrics

        points = np.array(
            [[x, y, z] for y, normal_y in ((-20.0, -1.0), (20.0, 1.0))
             for x in (-2.0, 2.0) for z in (-30.0, -10.0)],
            dtype=float,
        )
        normals = np.array(
            [[0.0, normal_y, 0.0] for y, normal_y in ((-20.0, -1.0), (20.0, 1.0))
             for _x in (-2.0, 2.0) for _z in (-30.0, -10.0)],
            dtype=float,
        )
        object_data = SimpleNamespace(
            points=points,
            normals=normals,
            center=np.zeros(3),
            radius=50.0,
            T_object_world=np.eye(4),
        )
        record = {
            "translation": [0.0, 0.0, 0.0],
            "rotation_matrix": np.eye(3).tolist(),
            "opening_mm": 50.0,
            "grasp_width_mm": 40.0,
        }

        metrics = compute_diagnostic_metrics(record, object_data)

        self.assertGreater(metrics["contact_points_left"], 0)
        self.assertGreater(metrics["contact_points_right"], 0)
        self.assertAlmostEqual(metrics["normal_alignment_robust"], 1.0)
        self.assertAlmostEqual(metrics["center_distance_normalized"], 0.0)
        self.assertAlmostEqual(metrics["stability_score"], 1.0)
        self.assertTrue(np.isfinite(metrics["diagnostic_score"]))


if __name__ == "__main__":
    unittest.main()
