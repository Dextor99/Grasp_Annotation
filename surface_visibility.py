"""Surface visibility filtering for a virtual viewing direction."""

import numpy as np


def _as_finite_vectors(values, name):
    vectors = np.asarray(values, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"{name} must contain only finite values")
    return vectors


def filter_front_facing_surface(points, normals, view_direction, min_dot=1e-8):
    """Filter surface samples whose normals face a virtual camera.

    ``view_direction`` points from the object to the virtual camera. A sample
    is retained iff its normalized normal dot the normalized view direction is
    greater than ``min_dot`` (default ``1e-8``). Returns the retained points,
    their normalized normals, and a boolean mask aligned with the input points.
    Zero-length normals are always excluded.
    """
    points = _as_finite_vectors(points, "points")
    normals = _as_finite_vectors(normals, "normals")
    if len(points) != len(normals):
        raise ValueError("points and normals must contain the same number of vectors")

    view_direction = np.asarray(view_direction, dtype=float)
    if view_direction.shape != (3,) or not np.all(np.isfinite(view_direction)):
        raise ValueError("view_direction must be a finite vector with shape (3,)")
    view_length = np.linalg.norm(view_direction)
    if view_length == 0.0:
        raise ValueError("view_direction must be non-zero")

    if not np.isscalar(min_dot) or not np.isfinite(min_dot):
        raise ValueError("min_dot must be a finite scalar")

    lengths = np.linalg.norm(normals, axis=1)
    nonzero_normals = lengths > 0.0
    normalized_normals = np.zeros_like(normals)
    normalized_normals[nonzero_normals] = (
        normals[nonzero_normals] / lengths[nonzero_normals, np.newaxis]
    )
    unit_view_direction = view_direction / view_length
    mask = nonzero_normals & (normalized_normals @ unit_view_direction > min_dot)

    return points[mask], normalized_normals[mask], mask
