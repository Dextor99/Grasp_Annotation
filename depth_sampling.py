"""Scale-normalized grasp approach depth sampling."""

import numpy as np


def generate_depth_samples(object_radius, num_depth=16, max_ratio=1.2):
    """Return monotonically increasing depths as fractions of object radius."""
    radius = float(object_radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("object_radius must be a positive finite number")
    if not isinstance(num_depth, int) or num_depth < 2:
        raise ValueError("num_depth must be an integer greater than one")
    max_ratio = float(max_ratio)
    if not np.isfinite(max_ratio) or max_ratio <= 0:
        raise ValueError("max_ratio must be a positive finite number")
    return np.linspace(0.0, radius * max_ratio, num=num_depth)
