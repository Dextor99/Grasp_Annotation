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


def ply_scale_to_meters(path: str | Path) -> float:
    """Return the fixed project conversion factor for a PLY path."""
    return float(get_model_scale(path)) / 1000.0


def normalized_p95(p95_m: float, extents_m: np.ndarray) -> float:
    extents = np.asarray(extents_m, dtype=float).reshape(-1)
    diameter = float(np.max(extents)) if extents.size else 0.0
    return float(p95_m) / diameter if diameter > 0 else float("inf")


def orient_normals_radially(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Orient normals consistently away from the point-cloud centroid."""
    points = np.asarray(points, dtype=float)
    result = np.asarray(normals, dtype=float).copy()
    if points.shape != result.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and normals must both have shape (N, 3)")
    center = points.mean(axis=0)
    inward = np.sum((points - center) * result, axis=1) < 0.0
    result[inward] *= -1.0
    return result


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


def reconstruct_one(*, name: str, ply_path: Path, output_root: Path, sdf_exe: Path | None = None) -> dict[str, Any]:
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
        orient_normals_radially(np.asarray(pcd.points), np.asarray(pcd.normals))
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
    vertices = np.asarray(loaded.vertices, dtype=float)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    bbox_min = points_m.min(axis=0)
    bbox_max = points_m.max(axis=0)
    ref_min = vertices.min(axis=0) if len(vertices) else np.zeros(3)
    ref_max = vertices.max(axis=0) if len(vertices) else np.zeros(3)
    # Uniform reference sampling makes the audit independent of mesh vertex
    # density while retaining the same deterministic reconstruction output.
    sampled = loaded.sample(min(10000, max(1000, len(points_m))))
    d_source_to_ref = _nearest_distances(points_m, sampled)
    d_ref_to_source = _nearest_distances(sampled, points_m)
    extents = bbox_max - bbox_min
    sdf_path = obj_path.with_suffix(".sdf")
    if sdf_exe is not None:
        subprocess.run([str(sdf_exe), str(obj_path), str(SDF_DIM), str(SDF_PADDING)], check=True)
    components = _face_component_count(loaded)
    report = {
        "object": name,
        "source_file": str(ply_path),
        "reference_file": str(obj_path),
        "reference_source": "reconstructed_from_surface_point_cloud",
        "scale_to_meter": scale,
        "source_bbox_min_m": bbox_min.tolist(), "source_bbox_max_m": bbox_max.tolist(),
        "reference_bbox_min_m": ref_min.tolist(), "reference_bbox_max_m": ref_max.tolist(),
        "source_extents_m": extents.tolist(), "reference_extents_m": (ref_max - ref_min).tolist(),
        "source_vertices": int(len(points_m)), "reference_vertices": int(len(vertices)),
        "reference_faces": int(len(faces)), "reference_connected_components": components,
        "watertight": bool(getattr(loaded, "is_watertight", False)),
        "surface_error_mean_m": float(np.mean(d_ref_to_source)),
        "surface_error_p95_m": float(np.percentile(d_ref_to_source, 95)),
        "surface_error_normalized_p95": normalized_p95(float(np.percentile(d_ref_to_source, 95)), extents),
        "source_to_reference_mean_m": float(np.mean(d_source_to_ref)),
        "sdf_dim": SDF_DIM, "sdf_padding": SDF_PADDING,
        "sdf_file": str(sdf_path), "sdf_exists": bool(sdf_path.is_file()),
        "gate_pass": bool(getattr(loaded, "is_watertight", False) and sdf_path.is_file()),
    }
    (output_dir / "asset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--sdf-exe", type=Path)
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
        reports.append(reconstruct_one(name=name, ply_path=ply, output_root=args.output_root, sdf_exe=args.sdf_exe))
    print(json.dumps({"objects": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
