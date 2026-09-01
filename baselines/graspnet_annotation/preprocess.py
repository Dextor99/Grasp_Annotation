"""Input geometry preparation for the independent GN-Full baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


UNIT_TO_METRES = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}


@dataclass(frozen=True)
class MeshReadiness:
    is_watertight: bool
    vertex_count: int
    face_count: int


@dataclass(frozen=True)
class LoadedMesh:
    mesh: trimesh.Trimesh
    source_path: Path
    input_unit: str
    scale_to_metres: float


def convert_vertices_to_meters(vertices: np.ndarray, input_unit: str) -> np.ndarray:
    """Convert a vertex array to metres without mutating the input."""

    if input_unit not in UNIT_TO_METRES:
        raise ValueError(f"input_unit must be one of {tuple(UNIT_TO_METRES)}, got {input_unit!r}")
    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("vertices must contain only finite values")
    return values * UNIT_TO_METRES[input_unit]


def _as_mesh(loaded: trimesh.Trimesh | trimesh.Scene, source_path: Path) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Mesh has no triangle geometry: {source_path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported geometry type {type(loaded).__name__}: {source_path}")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Mesh has no triangle geometry: {source_path}")
    return loaded


def validate_mesh_readiness(mesh_path: str | Path, sdf_path: str | Path | None, require_sdf: bool) -> MeshReadiness:
    """Validate the source mesh and, when requested, the presence of its SDF."""

    mesh_path = Path(mesh_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    if require_sdf and (sdf_path is None or not Path(sdf_path).is_file()):
        raise FileNotFoundError("SDF is required for Dex-Net force-closure; provide --sdf or generate one first")
    mesh = _as_mesh(trimesh.load_mesh(mesh_path, process=False), mesh_path)
    return MeshReadiness(bool(mesh.is_watertight), len(mesh.vertices), len(mesh.faces))


def load_mesh_in_metres(mesh_path: str | Path, input_unit: str) -> LoadedMesh:
    """Load a copy of ``mesh_path`` and convert its vertices to metres."""

    source_path = Path(mesh_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {source_path}")
    mesh = _as_mesh(trimesh.load_mesh(source_path, process=False), source_path).copy()
    mesh.vertices = convert_vertices_to_meters(mesh.vertices, input_unit)
    return LoadedMesh(mesh=mesh, source_path=source_path, input_unit=input_unit, scale_to_metres=UNIT_TO_METRES[input_unit])


def load_surface_ply_in_metres(ply_path: str | Path, input_unit: str) -> np.ndarray:
    """Load finite PLY surface points directly, without reconstructing a mesh."""
    import open3d as o3d

    source_path = Path(ply_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Surface PLY not found: {source_path}")
    cloud = o3d.io.read_point_cloud(str(source_path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Surface PLY has no valid points: {source_path}")
    return convert_vertices_to_meters(points, input_unit).astype(np.float32)
