"""Create a deterministic grasp-point-stratified GN force-closure subset.

The GN geometry run stores candidates in C-order as
``(grasp_point, view, angle, depth)``.  This module samples only geometry-valid
candidate ids, while distributing the requested budget as evenly as possible
over grasp points.  It deliberately does not score or alter any labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:  # Support both ``python -m`` and direct script execution.
    from .stratified_subset import build_sampling_manifest, sample_stratified_ids
except ImportError:  # pragma: no cover - exercised by the CLI invocation
    from stratified_subset import build_sampling_manifest, sample_stratified_ids


def stratified_sample_ids(
    collision: np.ndarray,
    *,
    target_size: int = 10_000,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample geometry-valid flattened ids stratified by grasp point.

    Parameters
    ----------
    collision:
        Boolean label tensor whose first dimension is the grasp-point stratum.
        ``False`` means geometry-valid, matching the GN label contract.
    target_size:
        Requested number of candidates.  If fewer valid candidates exist, all
        valid candidates are returned and the manifest reports that fact.
    seed:
        NumPy generator seed used independently within each point stratum.

    Returns
    -------
    selected_ids, details:
        Sorted, unique flattened C-order ids and JSON-ready sampling details
        (plus ``candidate_ids`` as an ndarray for callers writing artifacts).
    """
    return sample_stratified_ids(collision, target_count=target_size, seed=seed)


def write_sampling_artifacts(
    collision: np.ndarray,
    output: str | Path,
    *,
    target_size: int = 10_000,
    seed: int = 0,
    source_geometry_run: str | Path | None = None,
) -> dict[str, Any]:
    """Write sampled ids and a reproducibility manifest to ``output``."""
    selected, details = stratified_sample_ids(collision, target_size=target_size, seed=seed)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "sampled_candidate_ids.npy", selected, allow_pickle=False)
    manifest = build_sampling_manifest(details, source_geometry_run=source_geometry_run)
    (output_path / "sampling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    labels_path = args.geometry_run / "grasp_labels.npz"
    if not labels_path.is_file():
        raise FileNotFoundError(f"geometry labels not found: {labels_path}")
    with np.load(labels_path) as labels:
        if "collision" not in labels:
            raise KeyError(f"collision array missing from {labels_path}")
        details = write_sampling_artifacts(
            labels["collision"], args.output, target_size=args.target_size,
            seed=args.seed, source_geometry_run=args.geometry_run,
        )
    print({key: value for key, value in details.items() if key != "candidate_ids"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
