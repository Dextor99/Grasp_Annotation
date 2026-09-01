import json
import importlib.util
import unittest
from pathlib import Path


class GraspNetGateTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("graspnetAPI"), "requires the dedicated GN-Full environment")
    def test_debug_cube_gates_pass_with_official_backend(self):
        from baselines.graspnet_annotation.gates import report_dict, run_gates

        asset_dir = Path(__file__).parents[2] / "baselines" / "graspnet_annotation" / "assets" / "debug_cube"
        report = run_gates(asset_dir)
        self.assertTrue(report.gate2_topology)
        self.assertTrue(report.gate3_pose_convention)
        self.assertTrue(report.gate4_collision_geometry)
        self.assertTrue(report.sdf_load)
        self.assertEqual(report.candidate_count, 14_400)
        self.assertEqual(report.view_count, 300)
        self.assertEqual(report.sdf_shape, (64, 64, 64))
        # Keep the report JSON-safe; this catches accidental NumPy scalar leakage.
        json.dumps(report_dict(report))


if __name__ == "__main__":
    unittest.main()
