"""Evaluate frozen Ours grasps with the official GN/Dex-Net evaluator.

This module is intentionally an adapter, not a second grasp generator.  It
reads exported Ours records, converts their millimetre/object-frame pose to
the official metre/Dex-Net convention, and evaluates every record in its
input order.  The Ours V4 score is never replaced and is never used to
select the common-evaluation subset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.graspnet_annotation.config import DenseAnnotationConfig
from baselines.graspnet_annotation.grasp_point_sampling import sample_collision_points
from baselines.graspnet_annotation.official_adapter import (
    build_dexnet_grasp,
    build_force_closure_configs,
    evaluate_official_collision_with_dexgrasps,
    load_dexnet_model,
    score_official_force_closure_prepared,
)


# Ours uses x=gripper thickness, y=closing, z=approach (towards the object
# from the fingertip plane).  The official convention uses x=approach,
# y=closing, z=the remaining right-handed axis.  This maps columns as
# [ours_z, ours_y, -ours_x] and has determinant +1.
OURS_TO_OFFICIAL_AXES = np.asarray(
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    dtype=float,
)


def _finite_pose(record: dict) -> tuple[np.ndarray, np.ndarray]:
    try:
        translation = np.asarray(record["translation"], dtype=float)
        rotation = np.asarray(record["rotation_matrix"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid Ours pose: {exc}") from exc
    if translation.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("translation must be (3,) and rotation_matrix must be (3,3)")
    if not np.isfinite(translation).all() or not np.isfinite(rotation).all():
        raise ValueError("Ours pose must be finite")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("rotation_matrix must be a valid SO(3) matrix")
    return translation, rotation


def record_to_official_components(record: dict, T_object_world: np.ndarray | None = None) -> dict[str, object]:
    """Convert one exported Ours record to official pose components.

    ``translation`` is the Ours gripper origin after closure refinement.  The
    official API takes the surface point at the beginning of the approach and
    reconstructs the Dex-Net center as ``point + R[:, 0] * depth``.  Therefore
    the surface point is recovered by subtracting the Ours approach offset.
    """

    translation_object_mm, ours_rotation_object = _finite_pose(record)
    if T_object_world is None:
        object_world = np.eye(4, dtype=float)
    else:
        object_world = np.asarray(T_object_world, dtype=float)
        if object_world.shape != (4, 4) or not np.isfinite(object_world).all():
            raise ValueError("T_object_world must be a finite (4,4) matrix")
        if not np.allclose(object_world[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError("T_object_world must be homogeneous")
        object_rotation = object_world[:3, :3]
        if not np.allclose(object_rotation.T @ object_rotation, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(object_rotation), 1.0, atol=1e-5
        ):
            raise ValueError("T_object_world rotation must be a valid SO(3) matrix")
    try:
        depth_mm = float(record["depth_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid depth_mm: {exc}") from exc
    width_value = record.get("grasp_width_mm", record.get("opening_mm"))
    try:
        width_mm = float(width_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid grasp width: {exc}") from exc
    if not np.isfinite(depth_mm) or depth_mm <= 0:
        raise ValueError("depth_mm must be positive and finite")
    if not np.isfinite(width_mm) or width_mm <= 0:
        raise ValueError("grasp_width_mm/opening_mm must be positive and finite")

    # Exported Ours records are in the centered OBB object frame.  The
    # reference OBJ/SDF is in the original point-cloud/world frame, so apply
    # the same object-to-world transform used by ``prepare_object`` before
    # handing the pose to the official evaluator.
    translation_mm = object_world[:3, :3] @ translation_object_mm + object_world[:3, 3]
    ours_rotation = object_world[:3, :3] @ ours_rotation_object
    official_rotation = ours_rotation @ OURS_TO_OFFICIAL_AXES
    grasp_point_mm = translation_mm - ours_rotation[:, 2] * depth_mm
    center_m = translation_mm / 1000.0
    grasp_point_m = grasp_point_mm / 1000.0
    width_m = width_mm / 1000.0
    depth_m = depth_mm / 1000.0
    grasp_row = np.empty(17, dtype=np.float32)
    grasp_row[0] = 1.0
    grasp_row[1] = width_m
    grasp_row[2] = 0.02
    grasp_row[3] = depth_m
    grasp_row[4:13] = official_rotation.astype(np.float32).reshape(9)
    grasp_row[13:16] = center_m.astype(np.float32)
    grasp_row[16] = 0.0
    return {
        "grasp_point_m": grasp_point_m.astype(np.float32),
        "center_m": center_m.astype(np.float32),
        "rotation": official_rotation.astype(np.float32),
        "width_m": width_m,
        "depth_m": depth_m,
        "grasp_row": grasp_row,
    }


def summarize_common_evaluation(
    records: list[dict],
    collision: np.ndarray,
    empty: np.ndarray,
    mu_min: np.ndarray,
    scored_mask: np.ndarray,
    error_mask: np.ndarray,
    wall_time_s: float,
) -> dict:
    """Summarize common evaluation without changing input order or ranking."""

    n = len(records)
    arrays = {
        "collision": np.asarray(collision, dtype=bool).reshape(-1),
        "empty": np.asarray(empty, dtype=bool).reshape(-1),
        "mu_min": np.asarray(mu_min, dtype=float).reshape(-1),
        "scored_mask": np.asarray(scored_mask, dtype=bool).reshape(-1),
        "error_mask": np.asarray(error_mask, dtype=bool).reshape(-1),
    }
    if any(len(value) != n for value in arrays.values()):
        raise ValueError("all evaluation arrays must have one entry per record")
    geometry_valid = ~arrays["collision"] & ~arrays["empty"]
    fc_valid = geometry_valid & arrays["scored_mask"] & np.isfinite(arrays["mu_min"]) & (arrays["mu_min"] >= 0)
    scores = arrays["mu_min"][fc_valid]
    hq = scores <= 0.4
    native_scores = np.asarray(
        [float(record.get("score_total", 0.0)) for record in records], dtype=float
    )
    return {
        "n_candidates": int(n),
        "n_geometry_valid": int(geometry_valid.sum()),
        "geometry_valid_rate": float(geometry_valid.mean()) if n else 0.0,
        "common_eval_count": int(n),
        "common_fc_valid": int(fc_valid.sum()),
        # Use the full common-evaluation set as the primary denominator so
        # comparison tables have identical semantics for Ours and GN.  Keep
        # the geometry-conditional rate separately for diagnostics.
        "common_fc_valid_rate": float(fc_valid.sum() / n) if n else 0.0,
        "common_fc_valid_rate_geometry": float(fc_valid.sum() / geometry_valid.sum()) if geometry_valid.any() else 0.0,
        "n_mu_le_04": int(hq.sum()),
        "hq_rate_mu04": float(hq.mean()) if len(scores) else 0.0,
        "hq_yield": float(hq.sum() / n) if n else 0.0,
        "mean_mu": float(scores.mean()) if len(scores) else -1.0,
        "n_scored": int(arrays["scored_mask"].sum()),
        "n_errors": int(arrays["error_mask"].sum()),
        "n_collision": int(arrays["collision"].sum()),
        "n_empty": int(arrays["empty"].sum()),
        "common_eval_wall_time_s": float(wall_time_s),
        # This explicit marker makes it auditable that this adapter did not
        # re-rank Ours by common mu or discard records before evaluation.
        "native_score_order_preserved": True,
        "native_score_mean": float(native_scores.mean()) if n else 0.0,
    }


def load_reference_collision_points(reference_obj: str | Path, config: DenseAnnotationConfig) -> np.ndarray:
    """Deterministically build the official collision cloud from the OBJ."""

    import trimesh

    loaded = trimesh.load_mesh(str(reference_obj), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"reference OBJ contains no mesh geometry: {reference_obj}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"unsupported reference mesh type: {type(loaded)!r}")
    return sample_collision_points(mesh, config)


def evaluate_ours_records(
    records: list[dict],
    model_points_m: np.ndarray,
    sdf_prefix: str | Path,
    *,
    outlier_m: float = 0.05,
    empty_thresh: int = 10,
    friction_coefficients=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
    T_object_world: np.ndarray | None = None,
) -> tuple[dict, list[dict]]:
    """Run official collision and Dex-Net FC for every Ours record."""

    if not isinstance(records, list):
        raise ValueError("records must be a list")
    components = [record_to_official_components(record, T_object_world) for record in records]
    rows = np.stack([component["grasp_row"] for component in components], axis=0) if components else np.empty((0, 17), dtype=np.float32)
    points = np.asarray(model_points_m, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("model_points_m must be a non-empty finite (N, 3) array")
    started = time.perf_counter()
    if len(records):
        collision, empty, dexgrasps = evaluate_official_collision_with_dexgrasps(
            rows,
            points,
            points,
            outlier_m=outlier_m,
            empty_thresh=empty_thresh,
        )
    else:
        collision = np.empty(0, dtype=bool)
        empty = np.empty(0, dtype=bool)
        dexgrasps = []
    collision = np.asarray(collision, dtype=bool).reshape(-1)
    empty = np.asarray(empty, dtype=bool).reshape(-1)
    if len(collision) != len(records) or len(empty) != len(records):
        raise RuntimeError("official collision evaluator returned an unexpected result length")

    mu_min = np.full(len(records), -1.0, dtype=np.float32)
    scored_mask = np.zeros(len(records), dtype=bool)
    error_mask = np.zeros(len(records), dtype=bool)
    error_messages = []
    if len(records):
        dex_model = load_dexnet_model(sdf_prefix)
        fc_list, fc_configs = build_force_closure_configs(friction_coefficients)
        geometry_valid = ~collision & ~empty
        for index in np.flatnonzero(geometry_valid):
            try:
                # Rebuild from the explicit conversion rather than relying on
                # any common-evaluator ranking or score embedded in the row.
                component = components[int(index)]
                grasp = build_dexnet_grasp(
                    component["grasp_point_m"],
                    component["rotation"],
                    component["depth_m"],
                    component["width_m"],
                )
                value = score_official_force_closure_prepared(
                    grasp, dex_model, fc_list, fc_configs
                )
                if not np.isfinite(value):
                    raise ValueError(f"official force-closure score is non-finite: {value!r}")
                mu_min[int(index)] = float(value)
                scored_mask[int(index)] = True
            except Exception as exc:  # preserve per-record auditability
                error_mask[int(index)] = True
                error_messages.append({"index": int(index), "error": repr(exc)})
    elapsed = time.perf_counter() - started
    summary = summarize_common_evaluation(
        records, collision, empty, mu_min, scored_mask, error_mask, elapsed
    )
    summary["errors"] = error_messages
    output_records = []
    for index, record in enumerate(records):
        output_records.append(
            {
                "index": int(index),
                "native_score_total": record.get("score_total"),
                "native_score_total_v4": record.get("score_total_v4"),
                "collision": bool(collision[index]),
                "empty": bool(empty[index]),
                "geometry_valid": bool((~collision[index]) and (~empty[index])),
                "mu_min": float(mu_min[index]),
                "scored": bool(scored_mask[index]),
                "error": bool(error_mask[index]),
            }
        )
    return summary, output_records


def _write_outputs(output: Path, summary: dict, rows: list[dict], records: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    rows_path = output / "common_records.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    n = len(rows)
    np.savez_compressed(
        output / "common_eval.npz",
        candidate_ids=np.arange(n, dtype=np.int64),
        collision=np.asarray([row["collision"] for row in rows], dtype=bool),
        empty=np.asarray([row["empty"] for row in rows], dtype=bool),
        geometry_valid=np.asarray([row["geometry_valid"] for row in rows], dtype=bool),
        mu_min=np.asarray([row["mu_min"] for row in rows], dtype=np.float32),
        scored_mask=np.asarray([row["scored"] for row in rows], dtype=bool),
        error_mask=np.asarray([row["error"] for row in rows], dtype=bool),
    )
    # Keep the source records untouched, but copy them for convenient paired
    # inspection and to guarantee that common evaluation can never overwrite
    # Ours ranking fields.
    (output / "ours_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-results", type=Path, required=True, help="directory containing Ours grasps.json")
    parser.add_argument("--reference-obj", type=Path, required=True, help="same-object repaired/reference OBJ")
    parser.add_argument("--sdf-prefix", type=Path, required=True, help="official OBJ/SDF prefix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-points", type=Path, help="optional .npy collision cloud in metres")
    parser.add_argument(
        "--ours-object",
        type=Path,
        help="optional Ours input PLY; used to recover its object-to-world OBB transform (defaults to meta.json)",
    )
    parser.add_argument("--surface-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--empty-thresh", type=int, default=10)
    parser.add_argument("--outlier-m", type=float, default=0.05)
    args = parser.parse_args(argv)

    records_path = args.ours_results / "grasps.json"
    if not records_path.is_file():
        raise FileNotFoundError(f"missing Ours records: {records_path}")
    records = json.loads(records_path.read_text(encoding="utf-8"))
    meta_path = args.ours_results / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    ours_object = args.ours_object
    if ours_object is None and meta.get("object"):
        ours_object = Path(str(meta["object"]))
    object_transform = np.eye(4, dtype=float)
    if ours_object is not None:
        if not ours_object.is_file():
            candidate = PROJECT_ROOT / ours_object
            if candidate.is_file():
                ours_object = candidate
        if not ours_object.is_file():
            raise FileNotFoundError(
                f"Ours source object is required to align object-frame poses: {ours_object}"
            )
        # Import lazily so pure conversion/statistics remain usable without
        # Open3D and so the adapter does not affect Ours generation.
        from object_preprocess import prepare_object

        object_transform = np.asarray(prepare_object(str(ours_object)).T_object_world, dtype=float)
    config = DenseAnnotationConfig.full(surface_samples=args.surface_samples, seed=args.seed)
    if args.model_points:
        model_points = np.load(args.model_points)
    else:
        model_points = load_reference_collision_points(args.reference_obj, config)
    summary, rows = evaluate_ours_records(
        records,
        model_points,
        args.sdf_prefix,
        outlier_m=args.outlier_m,
        empty_thresh=args.empty_thresh,
        friction_coefficients=config.friction_coefficients,
        T_object_world=object_transform,
    )
    summary.update(
        {
            "method": "Ours-v1.2-common",
            "source_ours_results": str(args.ours_results),
            "reference_obj": str(args.reference_obj),
            "sdf_prefix": str(args.sdf_prefix),
            "model_point_count": int(len(model_points)),
            "config": config.to_dict(),
            "evaluator": "official_graspnet_collision_and_dexnet_force_closure",
            "selection_policy": "evaluate_all_ours_records_in_input_order",
            "ours_object": str(ours_object) if ours_object is not None else None,
            "T_object_world": object_transform.tolist(),
        }
    )
    summary["native_wall_time_s"] = float(meta.get("timings", {}).get("total_s", 0.0))
    _write_outputs(args.output, summary, rows, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
