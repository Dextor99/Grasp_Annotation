"""Virtual camera viewpoint sampling utilities."""

import math

import numpy as np


def generate_viewpoints(num_views=60):
    """Return ``num_views`` evenly distributed unit vectors on a sphere.

    The vectors are sampled using a Fibonacci sphere construction.
    """
    if isinstance(num_views, bool) or not isinstance(num_views, (int, np.integer)):
        raise ValueError("num_views must be a positive integer")
    if num_views <= 0:
        raise ValueError("num_views must be a positive integer")

    indices = np.arange(num_views, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / num_views
    radius = np.sqrt(1.0 - z * z)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    azimuth = indices * golden_angle

    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))
