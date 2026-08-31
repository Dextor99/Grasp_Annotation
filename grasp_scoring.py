"""Adapters that apply the existing grasp quality metrics to candidate grasps."""

from __future__ import annotations

from numbers import Real

import numpy as np

from grasp_score_V3 import compute_grasp_scores_simple


def _finite_score(value):
    """Return a plain finite float, or ``None`` for unavailable metrics."""
    if not isinstance(value, Real):
        return None
    score = float(value)
    return score if np.isfinite(score) else None


def score_grasp_candidates(object_data, grasps):
    """Score candidates with the legacy V3 metrics and rank best first.

    ``score_total`` deliberately does not introduce a new quality equation: it
    uses the existing force-closure score when finite and otherwise falls back
    to the existing inner-point ratio.  A zero fallback guarantees a finite
    sortable value even for incomplete diagnostic candidates.
    """
    if not hasattr(object_data, "cloud_down"):
        raise ValueError("object_data must provide cloud_down for grasp scoring")

    scored = compute_grasp_scores_simple(
        [dict(grasp) for grasp in grasps],
        object_data.cloud_down,
        vis=False,
    )
    for grasp in scored:
        total = _finite_score(grasp.get("score_force_closure"))
        if total is None:
            total = _finite_score(grasp.get("score_inner_points_ratio"))
        grasp["score_total"] = 0.0 if total is None else total

    return sorted(scored, key=lambda grasp: grasp["score_total"], reverse=True)
