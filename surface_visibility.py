"""Normal-based front-facing surface selection for virtual views."""

from __future__ import annotations

import numpy as np


def select_front_facing_surface(points, normals, view_direction, threshold=0.0):
    """Select points whose outward normal faces the camera direction.

    ``view_direction`` follows the project convention: object center to camera.
    This is a normal filter only; it does not claim to resolve self-occlusion.
    """
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    view = np.asarray(view_direction, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if normals.shape != points.shape:
        raise ValueError("normals must match points shape")
    norm = np.linalg.norm(view)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("view_direction must be a non-zero finite vector")
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    scores = normals @ (view / norm)
    mask = scores > threshold
    return points[mask], normals[mask]
