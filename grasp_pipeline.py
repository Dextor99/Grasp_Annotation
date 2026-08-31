"""Final end-to-end 6-DoF grasp annotation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from grasp_config import GraspGenerationConfig
from grasp_determinism import configure_determinism
from grasp_merge import merge_grasp_candidates
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
    unique_candidates = merge_grasp_candidates(
        scored_candidates,
        translation_threshold_mm=config.translation_merge_mm,
        rotation_threshold_deg=config.rotation_merge_deg,
    )
    timings["merge_s"] = perf_counter() - stage_start

    stage_start = perf_counter()
    raw_records = [normalize_grasp_record(grasp) for grasp in scored_candidates]
    unique_records = [normalize_grasp_record(grasp) for grasp in unique_candidates]
    timings["normalization_s"] = perf_counter() - stage_start
    timings["total_s"] = perf_counter() - total_start

    meta = {
        "object": str(Path(object_path)),
        "units": "mm",
        "input_scale_to_mm": float(object_data.scale),
        "point_count": int(len(object_data.points)),
        "raw_grasp_count": len(raw_records),
        "unique_grasp_count": len(unique_records),
        "config": config.to_dict(),
        "timings": timings,
    }
    return GraspAnnotationResult(
        raw_grasps=raw_records,
        unique_grasps=unique_records,
        meta=meta,
    )
