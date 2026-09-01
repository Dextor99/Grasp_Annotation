"""Safe, deterministic export for the independent baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import DenseAnnotationConfig
from .label_arrays import RawLabelArrays


_OUTPUT_FILES = {"grasp_labels.npz", "valid_grasps.npy", "summary.json", "timing.csv", "run_config.json"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def export_annotation_run(
    output: str | Path,
    labels: RawLabelArrays,
    summary: dict[str, Any],
    timing_rows: Iterable[dict[str, Any]],
    config: DenseAnnotationConfig | None = None,
) -> dict[str, Path]:
    output = Path(output)
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"Output path is not a directory: {output}")
        existing = {path.name for path in output.iterdir()}
        if existing:
            raise FileExistsError(f"Output directory must be new or empty: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    safe_summary = _json_safe(dict(summary))
    if config is not None:
        safe_summary.setdefault("config", config.to_dict())
    raw_count = int(np.prod(labels.collision.shape))
    valid_count = int(np.count_nonzero((~labels.collision) & (labels.scores >= 0.0)))
    safe_summary.setdefault("raw_candidate_count", raw_count)
    safe_summary.setdefault("valid_count", valid_count)
    safe_summary.setdefault("units", "m")
    summary_text = json.dumps(safe_summary, ensure_ascii=False, indent=2, allow_nan=False)
    run_config = {
        "config": config.to_dict() if config is not None else safe_summary.get("config", {}),
        "parameter_provenance": config.parameter_provenance() if config is not None else {},
    }
    run_config_text = json.dumps(_json_safe(run_config), ensure_ascii=False, indent=2, allow_nan=False)
    rows = [_json_safe(dict(row)) for row in timing_rows]
    fields = sorted({key for row in rows for key in row}) or ["stage", "seconds"]

    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "grasp_labels.npz", points=labels.points, offsets=labels.offsets, collision=labels.collision, scores=labels.scores)
    np.save(output / "valid_grasps.npy", labels.to_valid_grasps(config) if config is not None else np.empty((0, 17), dtype=np.float32), allow_pickle=False)
    (output / "summary.json").write_text(summary_text, encoding="utf-8")
    (output / "run_config.json").write_text(run_config_text, encoding="utf-8")
    with (output / "timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {name.removesuffix(".json").removesuffix(".npz").removesuffix(".npy").removesuffix(".csv"): output / name for name in _OUTPUT_FILES}
