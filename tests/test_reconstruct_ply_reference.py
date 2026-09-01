import numpy as np

from scripts.common_eval.reconstruct_ply_reference import (
    normalized_p95,
    orient_normals_radially,
    ply_scale_to_meters,
)


def test_ply_scale_uses_project_units():
    assert ply_scale_to_meters("model/juxing.ply") == 1e-3
    assert ply_scale_to_meters("model/colmap/cat.ply") == 1.0


def test_normalized_p95_uses_object_diameter():
    assert np.isclose(normalized_p95(0.02, np.array([1.0, 2.0, 3.0])), 0.02 / 3.0)


def test_radial_orientation_flips_inward_normals():
    points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    normals = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = orient_normals_radially(points, normals)
    np.testing.assert_allclose(result, [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
