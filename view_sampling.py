"""Uniform virtual-view direction sampling."""

from __future__ import annotations

import math
import numpy as np


def fibonacci_directions(num_views: int) -> np.ndarray:
    """Return ``num_views`` unit directions on the sphere."""
    if not isinstance(num_views, int) or num_views < 1:
        raise ValueError("num_views must be a positive integer")
    if num_views == 1:
        # A deterministic non-polar direction is more useful than a sphere
        # pole for sparse point clouds, whose normals may not cover the pole.
        return np.array([[0.0, 0.0, 1.0]])
    # Midpoint sampling avoids exact polar directions, which often have no
    # front-facing points in finite point clouds.
    indices = np.arange(num_views, dtype=float) + 0.5
    y = 1.0 - 2.0 * indices / num_views
    radial = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden_angle * indices
    directions = np.column_stack((np.cos(theta) * radial, y, np.sin(theta) * radial))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)
