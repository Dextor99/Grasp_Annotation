import tempfile
import unittest
from pathlib import Path

import numpy as np


class GraspNetPreprocessTests(unittest.TestCase):
    def test_converts_millimetre_vertices_to_metres_once(self):
        from baselines.graspnet_annotation.preprocess import convert_vertices_to_meters

        vertices = np.array([[0.0, 0.0, 0.0], [100.0, 20.0, 10.0]])
        converted = convert_vertices_to_meters(vertices, "mm")
        np.testing.assert_allclose(converted[1], [0.1, 0.02, 0.01])

    def test_rejects_missing_sdf_for_force_closure_ready_run(self):
        from baselines.graspnet_annotation.preprocess import validate_mesh_readiness

        with tempfile.TemporaryDirectory() as temporary_directory:
            mesh = Path(temporary_directory) / "object.stl"
            mesh.write_bytes(b"solid empty\nendsolid empty\n")
            with self.assertRaisesRegex(FileNotFoundError, "SDF"):
                validate_mesh_readiness(mesh, sdf_path=None, require_sdf=True)

    def test_loads_surface_ply_without_mesh_reconstruction(self):
        from baselines.graspnet_annotation.preprocess import load_surface_ply_in_metres

        ply = Path(__file__).parents[2] / "model" / "juxing.ply"
        points = load_surface_ply_in_metres(ply, "mm")
        self.assertEqual(points.ndim, 2)
        self.assertEqual(points.shape[1], 3)
        self.assertGreater(len(points), 0)
        self.assertLessEqual(float(points.max()), 0.2 + 1e-5)


if __name__ == "__main__":
    unittest.main()
