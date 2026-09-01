"""Deterministically reconstruct evaluation-only meshes from frozen PLY input.

The reconstructed OBJ/SDF is used only by GN-style collision and Dex-Net
evaluation.  It is never fed back into Ours generation and is never labelled
as ground-truth geometry.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from model_scale import get_model_scale


POISSON_DEPTH = 8
POISSON_SCALE = 1.1
NORMAL_K = 30
SDF_DIM = 100
SDF_PADDING = 5
AUDIT_SAMPLE_SEED = 0
MAX_NORMALIZED_P95 = 0.02
MAX_BBOX_RELATIVE_ERROR = 0.05


def ply_scale_to_meters(path: str | Path) -> float:
    """Return the fixed project conversion factor for a PLY path."""
    return float(get_model_scale(path)) / 1000.0


def normalized_p95(p95_m: float, extents_m: np.ndarray) -> float:
    extents = np.asarray(extents_m, dtype=float).reshape(-1)
    diameter = float(np.max(extents)) if extents.size else 0.0
    return float(p95_m) / diameter if diameter > 0 else float("inf")


def orient_normals_global_sign(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Apply one global radial sign decision to an already coherent normal field.

    The tangent-plane pass establishes local consistency.  We only resolve its
    remaining global sign ambiguity here; individual points are never flipped,
    which would introduce discontinuities on thin or non-convex objects.
    """
    points = np.asarray(points, dtype=float)
    result = np.asarray(normals, dtype=float).copy()
    if points.shape != result.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and normals must both have shape (N, 3)")
    if len(points) == 0:
        return result
    center = points.mean(axis=0)
    radial_scores = np.sum((points - center) * result, axis=1)
    if float(np.median(radial_scores)) < 0.0:
        result *= -1.0
    return result


def orient_normals_radially(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Backward-compatible name for :func:`orient_normals_global_sign`."""
    return orient_normals_global_sign(points, normals)


def deterministic_surface_sample(mesh: Any, count: int, *, seed: int = AUDIT_SAMPLE_SEED) -> np.ndarray:
    """Sample a mesh surface reproducibly without leaking RNG state.

    ``trimesh`` uses NumPy's legacy global RNG internally.  Saving/restoring
    its state keeps the audit deterministic while avoiding side effects on the
    rest of a caller's process.
    """
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        sampled = mesh.sample(count)
    finally:
        np.random.set_state(state)
    return np.asarray(sampled, dtype=float)


def evaluate_reconstruction_gate(*, watertight: bool, connected_components: int,
                                 normalized_p95: float, source_bbox_min: np.ndarray,
                                 source_bbox_max: np.ndarray, reference_bbox_min: np.ndarray,
                                 reference_bbox_max: np.ndarray, sdf_exists: bool,
                                 official_dexnet_load_success: bool) -> tuple[dict[str, bool], bool]:
    """Evaluate the strict, object-independent reconstruction acceptance gate."""
    source_min = np.asarray(source_bbox_min, dtype=float).reshape(3)
    source_max = np.asarray(source_bbox_max, dtype=float).reshape(3)
    reference_min = np.asarray(reference_bbox_min, dtype=float).reshape(3)
    reference_max = np.asarray(reference_bbox_max, dtype=float).reshape(3)
    source_extents = source_max - source_min
    reference_extents = reference_max - reference_min
    denom = np.maximum(np.abs(source_extents), np.finfo(float).eps)
    extent_error = float(np.max(np.abs(reference_extents - source_extents) / denom))
    source_center = (source_min + source_max) / 2.0
    reference_center = (reference_min + reference_max) / 2.0
    diameter = max(float(np.max(np.abs(source_extents))), np.finfo(float).eps)
    center_error = float(np.max(np.abs(reference_center - source_center)) / diameter)
    bbox_error = max(extent_error, center_error)
    checks = {
        "watertight": bool(watertight),
        "connected_components": int(connected_components) == 1,
        "surface_error": bool(np.isfinite(normalized_p95) and normalized_p95 <= MAX_NORMALIZED_P95),
        "bbox": bool(np.isfinite(bbox_error) and bbox_error <= MAX_BBOX_RELATIVE_ERROR),
        "sdf": bool(sdf_exists),
        "official_dexnet_load": bool(official_dexnet_load_success),
    }
    return checks, bool(all(checks.values()))


def _nearest_distances(query_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    import open3d as o3d
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(reference_points, dtype=float)))
    tree = o3d.geometry.KDTreeFlann(source)
    distances = []
    for point in np.asarray(query_points, dtype=float):
        count, _, squared = tree.search_knn_vector_3d(point, 1)
        distances.append(float(np.sqrt(squared[0])) if count else float("inf"))
    return np.asarray(distances, dtype=float)


def _face_component_count(mesh: Any) -> int:
    """Count connected triangle components without trimesh/networkx repair."""
    faces = np.asarray(mesh.faces)
    if faces.ndim != 2 or len(faces) == 0:
        return 0
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency) == 0:
        return int(len(faces))
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    rows = np.concatenate([adjacency[:, 0], adjacency[:, 1], np.arange(len(faces))])
    cols = np.concatenate([adjacency[:, 1], adjacency[:, 0], np.arange(len(faces))])
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(faces), len(faces)))
    return int(connected_components(graph.tocsr(), directed=False, return_labels=False))


def dominant_component_face_mask(faces: np.ndarray, adjacency: np.ndarray,
                                 area_faces: np.ndarray, *, min_area_ratio: float = 0.99) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a largest-component mask only when it dominates surface area.

    This is a uniform post-reconstruction cleanup rule.  It never fills holes
    or changes a mesh whose largest component is below the area threshold.
    """
    if not 0.0 < float(min_area_ratio) <= 1.0:
        raise ValueError("min_area_ratio must be in (0, 1]")
    faces = np.asarray(faces, dtype=np.int64)
    adjacency = np.asarray(adjacency, dtype=np.int64)
    areas = np.asarray(area_faces, dtype=float).reshape(-1)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) != len(areas):
        raise ValueError("faces must be (N, 3) and area_faces must have length N")
    if len(faces) == 0:
        return np.zeros(0, dtype=bool), {"applied": False, "components": 0, "main_area_ratio": 0.0}
    if adjacency.size == 0:
        adjacency = np.empty((0, 2), dtype=np.int64)
    if adjacency.ndim != 2 or adjacency.shape[1] != 2:
        raise ValueError("adjacency must be (M, 2)")
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    rows = np.concatenate([adjacency[:, 0], adjacency[:, 1], np.arange(len(faces))])
    cols = np.concatenate([adjacency[:, 1], adjacency[:, 0], np.arange(len(faces))])
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(faces), len(faces)))
    components, labels = connected_components(graph.tocsr(), directed=False, return_labels=True)
    totals = np.bincount(labels, weights=areas, minlength=components)
    total_area = float(np.sum(areas))
    main_index = int(np.argmax(totals))
    main_ratio = float(totals[main_index] / total_area) if total_area > 0 else 0.0
    applied = bool(main_ratio >= float(min_area_ratio) and components > 1)
    mask = labels == main_index if applied else np.ones(len(faces), dtype=bool)
    return mask, {"applied": applied, "components": int(components), "main_area_ratio": main_ratio,
                  "removed_faces": int(np.count_nonzero(~mask))}


def remove_tiny_components(mesh: Any, *, min_area_ratio: float = 0.99) -> tuple[Any, dict[str, Any]]:
    """Drop disconnected components only when one component owns >=99% area."""
    import trimesh

    faces = np.asarray(mesh.faces, dtype=np.int64)
    mask, info = dominant_component_face_mask(
        faces, np.asarray(mesh.face_adjacency, dtype=np.int64),
        np.asarray(mesh.area_faces, dtype=float), min_area_ratio=min_area_ratio,
    )
    if not info["applied"]:
        return mesh, info
    kept_faces = faces[mask]
    used_vertices, remapped = np.unique(kept_faces.reshape(-1), return_inverse=True)
    cleaned = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices)[used_vertices],
        faces=remapped.reshape((-1, 3)), process=False,
    )
    return cleaned, info


def _official_dexnet_load_check(obj_path: Path, sdf_path: Path) -> tuple[bool, str | None]:
    """Check the official OBJ/SDF loader without masking backend failures."""
    if not sdf_path.is_file():
        return False, "sdf_missing"
    try:
        from baselines.graspnet_annotation.official_adapter import load_dexnet_model

        load_dexnet_model(obj_path.with_suffix(""))
    except Exception as exc:  # official backend availability is environment-dependent
        return False, repr(exc)
    return True, None


def reconstruct_one(*, name: str, ply_path: Path, output_root: Path, sdf_exe: Path | None = None,
                    remove_tiny_components_after_reconstruction: bool = False) -> dict[str, Any]:
    import open3d as o3d
    import trimesh

    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)
    cloud = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(cloud.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < NORMAL_K:
        raise ValueError(f"{ply_path} must contain at least {NORMAL_K} finite points")
    if not np.isfinite(points).all():
        raise ValueError(f"{ply_path} contains non-finite points")
    scale = ply_scale_to_meters(ply_path)
    points_m = points * scale
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_m))
    if cloud.has_normals() and len(cloud.normals) == len(points):
        normals = np.asarray(cloud.normals, dtype=float)
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_K))
    # A fixed tangent-plane orientation is used for every object.  No object-
    # specific Poisson parameters or manual hole filling are permitted.
    pcd.orient_normals_consistent_tangent_plane(NORMAL_K)
    pcd.normals = o3d.utility.Vector3dVector(
        orient_normals_global_sign(np.asarray(pcd.points), np.asarray(pcd.normals))
    )
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH, scale=POISSON_SCALE, linear_fit=True
    )
    mesh.compute_vertex_normals()
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = output_dir / "reference_reconstructed.obj"
    o3d.io.write_triangle_mesh(str(obj_path), mesh, write_vertex_normals=False, write_triangle_uvs=False)

    loaded = trimesh.load_mesh(obj_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    raw_components = _face_component_count(loaded)
    dominant_mesh, cleanup_info = remove_tiny_components(loaded)
    if remove_tiny_components_after_reconstruction and cleanup_info["applied"]:
        loaded = dominant_mesh
        loaded.export(obj_path)
    vertices = np.asarray(loaded.vertices, dtype=float)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    bbox_min = points_m.min(axis=0)
    bbox_max = points_m.max(axis=0)
    ref_min = vertices.min(axis=0) if len(vertices) else np.zeros(3)
    ref_max = vertices.max(axis=0) if len(vertices) else np.zeros(3)
    # Uniform reference sampling is seeded and isolated from the caller RNG.
    sample_count = min(10000, max(1000, len(points_m)))
    sampled = deterministic_surface_sample(loaded, sample_count, seed=AUDIT_SAMPLE_SEED)
    d_source_to_ref = _nearest_distances(points_m, sampled)
    d_ref_to_source = _nearest_distances(sampled, points_m)
    extents = bbox_max - bbox_min
    sdf_path = obj_path.with_suffix(".sdf")
    if sdf_exe is not None:
        subprocess.run([str(sdf_exe), str(obj_path), str(SDF_DIM), str(SDF_PADDING)], check=True)
    components = _face_component_count(loaded)
    sdf_exists = bool(sdf_path.is_file())
    official_load_ok, official_load_error = _official_dexnet_load_check(obj_path, sdf_path)
    reference_extents = ref_max - ref_min
    norm_p95 = normalized_p95(float(np.percentile(d_ref_to_source, 95)), extents)
    gate_checks, gate_pass = evaluate_reconstruction_gate(
        watertight=bool(getattr(loaded, "is_watertight", False)),
        connected_components=components,
        normalized_p95=norm_p95,
        source_bbox_min=bbox_min,
        source_bbox_max=bbox_max,
        reference_bbox_min=ref_min,
        reference_bbox_max=ref_max,
        sdf_exists=sdf_exists,
        official_dexnet_load_success=official_load_ok,
    )
    report = {
        "object": name,
        "source_file": str(ply_path),
        "reference_file": str(obj_path),
        "reference_source": "reconstructed_from_surface_point_cloud",
        "scale_to_meter": scale,
        "source_bbox_min_m": bbox_min.tolist(), "source_bbox_max_m": bbox_max.tolist(),
        "reference_bbox_min_m": ref_min.tolist(), "reference_bbox_max_m": ref_max.tolist(),
        "source_extents_m": extents.tolist(), "reference_extents_m": reference_extents.tolist(),
        "source_vertices": int(len(points_m)), "reference_vertices": int(len(vertices)),
        "reference_faces": int(len(faces)), "reference_connected_components": components,
        "raw_reference_connected_components": int(raw_components),
        "tiny_component_cleanup": cleanup_info,
        "watertight": bool(getattr(loaded, "is_watertight", False)),
        "surface_error_mean_m": float(np.mean(d_ref_to_source)),
        "surface_error_p95_m": float(np.percentile(d_ref_to_source, 95)),
        "surface_error_normalized_p95": norm_p95,
        "source_to_reference_mean_m": float(np.mean(d_source_to_ref)),
        "sdf_dim": SDF_DIM, "sdf_padding": SDF_PADDING,
        "audit_sample_count": int(sample_count), "audit_sample_seed": AUDIT_SAMPLE_SEED,
        "bbox_relative_error": float(max(
            np.max(np.abs(reference_extents - extents) / np.maximum(np.abs(extents), np.finfo(float).eps)),
            np.max(np.abs(((ref_min + ref_max) - (bbox_min + bbox_max)) / 2.0)
                   / max(float(np.max(np.abs(extents))), np.finfo(float).eps)),
        )),
        "sdf_file": str(sdf_path), "sdf_exists": sdf_exists,
        "official_dexnet_load_success": official_load_ok,
        "official_dexnet_load_error": official_load_error,
        "gate_checks": gate_checks,
        "gate_pass": gate_pass,
    }
    (output_dir / "asset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--sdf-exe", type=Path)
    parser.add_argument("--remove-tiny-components", action="store_true",
                        help="uniformly retain the dominant component only when it owns >=99%% of surface area")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = {str(item["name"]): item for item in manifest.get("objects", [])}
    names = args.objects or list(entries)
    reports = []
    for name in names:
        if name not in entries:
            raise ValueError(f"object {name!r} is not present in manifest")
        ply = Path(entries[name]["ours_ply"])
        if not ply.is_absolute():
            ply = Path.cwd() / ply
        reports.append(reconstruct_one(
            name=name, ply_path=ply, output_root=args.output_root, sdf_exe=args.sdf_exe,
            remove_tiny_components_after_reconstruction=args.remove_tiny_components,
        ))
    print(json.dumps({"objects": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
