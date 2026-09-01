import numpy as np

from scripts.common_eval.reconstruct_ply_reference import (
    deterministic_surface_sample,
    dominant_component_face_mask,
    evaluate_reconstruction_gate,
    normalized_p95,
    orient_normals_global_sign,
    ply_scale_to_meters,
)


def test_ply_scale_uses_project_units():
    assert ply_scale_to_meters("model/juxing.ply") == 1e-3
    assert ply_scale_to_meters("model/colmap/cat.ply") == 1.0


def test_normalized_p95_uses_object_diameter():
    assert np.isclose(normalized_p95(0.02, np.array([1.0, 2.0, 3.0])), 0.02 / 3.0)


def test_global_orientation_flips_all_normals_from_one_median_sign():
    points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    normals = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = orient_normals_global_sign(points, normals)
    np.testing.assert_allclose(result, [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])


def test_global_orientation_does_not_flip_individual_outliers():
    points = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    normals = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    result = orient_normals_global_sign(points, normals)
    np.testing.assert_allclose(result, normals)


def test_surface_audit_sampling_is_reproducible():
    class SampleOnlyMesh:
        def sample(self, count):
            return np.random.random((count, 3))

    mesh = SampleOnlyMesh()
    first = deterministic_surface_sample(mesh, 128, seed=17)
    second = deterministic_surface_sample(mesh, 128, seed=17)
    np.testing.assert_array_equal(first, second)


def test_reconstruction_gate_requires_all_strict_checks():
    checks, passed = evaluate_reconstruction_gate(
        watertight=True,
        connected_components=1,
        normalized_p95=0.01,
        source_bbox_min=np.zeros(3),
        source_bbox_max=np.ones(3),
        reference_bbox_min=np.zeros(3),
        reference_bbox_max=np.ones(3) * 1.01,
        sdf_exists=True,
        official_dexnet_load_success=True,
    )
    assert passed is True
    assert all(checks.values())

    checks, passed = evaluate_reconstruction_gate(
        watertight=True,
        connected_components=1,
        normalized_p95=0.01,
        source_bbox_min=np.zeros(3),
        source_bbox_max=np.ones(3),
        reference_bbox_min=np.zeros(3),
        reference_bbox_max=np.ones(3) * 1.01,
        sdf_exists=True,
        official_dexnet_load_success=False,
    )
    assert passed is False
    assert checks["official_dexnet_load"] is False


def test_dominant_component_cleanup_is_uniform_and_area_gated():
    faces = np.array([[0, 1, 2], [2, 1, 3], [4, 5, 6]])
    adjacency = np.array([[0, 1]])
    area_faces = np.array([0.50, 0.49, 0.01])
    mask, info = dominant_component_face_mask(faces, adjacency, area_faces, min_area_ratio=0.99)
    np.testing.assert_array_equal(mask, [True, True, False])
    assert info["applied"] is True
    assert np.isclose(info["main_area_ratio"], 0.99)

    mask, info = dominant_component_face_mask(faces, adjacency, area_faces, min_area_ratio=0.995)
    np.testing.assert_array_equal(mask, [True, True, True])
    assert info["applied"] is False
