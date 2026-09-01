"""Executable Gate 2--4 checks for the independent GN-Full baseline.

The gates are intentionally small and deterministic.  They validate the
official candidate topology and pose convention before any expensive SDF or
force-closure evaluation is attempted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .candidate_generation import iter_candidate_batches
from .config import DenseAnnotationConfig
from .gripper_geometry import evaluate_gripper_geometry
from .official_adapter import evaluate_official_collision, load_dexnet_model
from .preprocess import load_mesh_in_metres, validate_mesh_readiness
from .view_sampling import generate_view_rotations, generate_views


@dataclass(frozen=True)
class GateReport:
    gate2_topology: bool
    gate3_pose_convention: bool
    gate4_collision_geometry: bool
    sdf_load: bool
    candidate_count: int
    view_count: int
    sdf_shape: tuple[int, ...] | None
    details: dict


def _synthetic_grasp() -> np.ndarray:
    grasp = np.zeros((1, 17), dtype=np.float32)
    grasp[:, 0] = 1.0
    grasp[:, 1] = 0.06  # opening
    grasp[:, 2] = 0.02  # height
    grasp[:, 3] = 0.02  # depth
    grasp[:, 4:13] = np.eye(3, dtype=np.float32).reshape(-1)
    return grasp


def _synthetic_scene(kind: str) -> np.ndarray:
    x = np.linspace(-0.005, 0.015, 5)
    y = np.linspace(-0.025, 0.025, 5)
    points = np.asarray([[a, b, 0.0] for a in x for b in y], dtype=np.float32)
    if kind == "empty":
        return points + np.array([0.0, 0.1, 0.0], dtype=np.float32)
    if kind == "penetrating":
        return np.vstack((points, np.array([[0.0, 0.035, 0.0]], dtype=np.float32)))
    if kind != "good":
        raise ValueError(f"unknown synthetic scene kind: {kind}")
    return points


def run_gates(debug_asset_dir: str | Path, config: DenseAnnotationConfig | None = None) -> GateReport:
    """Run topology, pose, collision, and paired OBJ/SDF readiness gates."""

    config = config or DenseAnnotationConfig.full()
    point = np.zeros((1, 3), dtype=np.float32)
    candidate_count = sum(batch.size for batch in iter_candidate_batches(point, config, point_batch_size=1))
    gate2 = candidate_count == config.candidates_per_point == 14_400

    views = generate_views(config.num_views)
    rotations = generate_view_rotations(views, np.zeros(config.num_views, dtype=np.float32))
    orthogonal = np.allclose(np.matmul(rotations.transpose(0, 2, 1), rotations), np.eye(3), atol=1e-5)
    proper = np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5)
    gate3 = views.shape == (300, 3) and rotations.shape == (300, 3, 3) and orthogonal and proper

    good = _synthetic_scene("good")
    empty = _synthetic_scene("empty")
    penetrating = _synthetic_scene("penetrating")
    expected_official = {
        "good": (False, False),
        "empty": (True, True),
        "penetrating": (True, False),
    }
    expected_geometry = {
        "good": (False, False),
        "empty": (False, True),
        "penetrating": (True, False),
    }
    geometry_results = {}
    official_results = {}
    for kind, scene in (("good", good), ("empty", empty), ("penetrating", penetrating)):
        local = evaluate_gripper_geometry(
            scene, np.zeros(3), np.eye(3), opening_m=0.06, depth_m=0.02,
            height_m=0.02, depth_base_m=0.02, finger_width_m=0.01, empty_thresh=10,
        )
        local_pair = (bool(local.collision), bool(local.empty))
        collision, empty_mask = evaluate_official_collision(_synthetic_grasp(), scene, scene)
        official_pair = (bool(collision[0]), bool(empty_mask[0]))
        geometry_results[kind] = {"collision": local_pair[0], "empty": local_pair[1], "inner_count": local.inner_count}
        official_results[kind] = {"collision": official_pair[0], "empty": official_pair[1]}
    gate4 = all(
        (value["collision"], value["empty"]) == expected_official[kind]
        for kind, value in official_results.items()
    ) and all(
        (value["collision"], value["empty"]) == expected_geometry[kind]
        for kind, value in geometry_results.items()
    )

    asset_dir = Path(debug_asset_dir)
    obj_path = asset_dir / "debug_cube.obj"
    sdf_path = asset_dir / "debug_cube.sdf"
    readiness = validate_mesh_readiness(obj_path, sdf_path=sdf_path, require_sdf=True)
    loaded = load_mesh_in_metres(obj_path, input_unit="m")
    dexnet = load_dexnet_model(asset_dir / "debug_cube")
    sdf_shape = tuple(int(v) for v in dexnet.sdf.data.shape)
    sdf_ok = readiness.face_count > 0 and readiness.is_watertight and loaded.mesh.vertices.shape == (8, 3)

    return GateReport(
        gate2_topology=gate2,
        gate3_pose_convention=gate3,
        gate4_collision_geometry=gate4,
        sdf_load=sdf_ok,
        candidate_count=int(candidate_count),
        view_count=int(len(views)),
        sdf_shape=sdf_shape,
        details={"geometry": geometry_results, "official_collision": official_results},
    )


def report_dict(report: GateReport) -> dict:
    return asdict(report)
