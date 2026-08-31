"""Final end-to-end 6-DoF grasp annotation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from grasp_config import GraspGenerationConfig
from grasp_determinism import configure_determinism
from grasp_merge import merge_grasp_candidates
from grasp_refinement import refine_grasp_closures, validate_refined_grasp_closures
from grasp_schema import normalize_grasp_record
from grasp_scoring import score_grasp_candidates
from multi_view_grasp import generate_multi_view_grasps
from object_preprocess import prepare_object


@dataclass
class GraspAnnotationResult:
    raw_grasps: list
    unique_grasps: list
    meta: dict


def run_grasp_annotation(object_path, config=None):
    """Run the frozen prepare/generate/score/merge/normalize method chain."""
    config = config or GraspGenerationConfig()
    total_start = perf_counter()
    timings = {}

    stage_start = perf_counter()
    configure_determinism(config.deterministic, config.random_seed)
    timings["determinism_s"] = perf_counter() - stage_start

    stage_start = perf_counter()
    object_data = prepare_object(object_path)
    timings["prepare_s"] = perf_counter() - stage_start

    stage_start = perf_counter()
    raw_candidates = generate_multi_view_grasps(
        object_path,
        object_data=object_data,
        config=config,
    )
    timings["generation_s"] = perf_counter() - stage_start

    stage_start = perf_counter()
    scored_candidates = score_grasp_candidates(object_data, raw_candidates)
    timings["scoring_s"] = perf_counter() - stage_start

    stage_start = perf_counter()
    refined_candidates = refine_grasp_closures(
        scored_candidates,
        margin_mm=config.closure_margin_mm,
    )
    timings["refinement_s"] = perf_counter() - stage_start

    stage_start = perf_counter()
    validated_candidates = validate_refined_grasp_closures(
        refined_candidates,
        point_cloud=object_data.cloud_down,
        T_object_world=object_data.T_object_world,
        threshold_mm=config.collision_threshold_mm,
    )
    timings["closure_validation_s"] = perf_counter() - stage_start

    closure_geometry_rejected = sum(
        not candidate.get("closure_geometry_valid", True)
        for candidate in refined_candidates
    )
    closure_pose_collision_rejected = (
        len(refined_candidates) - closure_geometry_rejected - len(validated_candidates)
    )

    stage_start = perf_counter()
    unique_candidates = merge_grasp_candidates(
        validated_candidates,
        translation_threshold_mm=config.translation_merge_mm,
        rotation_threshold_deg=config.rotation_merge_deg,
    )
    timings["merge_s"] = perf_counter() - stage_start

    stage_start = perf_counter()
    raw_records = [normalize_grasp_record(grasp) for grasp in validated_candidates]
    unique_records = [normalize_grasp_record(grasp) for grasp in unique_candidates]
    timings["normalization_s"] = perf_counter() - stage_start
    timings["total_s"] = perf_counter() - total_start

    raw_count = len(raw_records)
    unique_count = len(unique_records)
    candidate_counts = {
        "generated_candidate_count": len(raw_candidates),
        "scored_candidate_count": len(scored_candidates),
        "refinement_input_count": len(refined_candidates),
        "closure_geometry_rejected": int(closure_geometry_rejected),
        "closure_pose_collision_rejected": int(closure_pose_collision_rejected),
        "closure_valid_count": len(validated_candidates),
        "unique_grasp_count": unique_count,
    }
    merge_reduction_ratio = (
        1.0 - unique_count / raw_count if raw_count else 0.0
    )
    meta = {
        "object": str(Path(object_path)),
        "units": "mm",
        "input_scale_to_mm": float(object_data.scale),
        "point_count": int(len(object_data.points)),
        "raw_grasp_count": raw_count,
        "unique_grasp_count": unique_count,
        # ``raw_grasp_count`` is retained for compatibility and means the
        # closure-valid, pre-merge set.  Use candidate_counts for experiments.
        "candidate_counts": candidate_counts,
        "closure_validation": {
            "input_count": len(refined_candidates),
            "geometry_rejected": int(closure_geometry_rejected),
            "pose_collision_rejected": int(closure_pose_collision_rejected),
            "valid_count": len(validated_candidates),
            "rejected_count": len(refined_candidates) - len(validated_candidates),
        },
        "merge_reduction_ratio": merge_reduction_ratio,
        "config": config.to_dict(),
        "timings": timings,
    }
    return GraspAnnotationResult(
        raw_grasps=raw_records,
        unique_grasps=unique_records,
        meta=meta,
    )
