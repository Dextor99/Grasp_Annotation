"""Sparse approach directions guided by a local outward surface normal."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ApproachSample:
    approach_id: int
    direction: np.ndarray
    offset_deg: float


def sample_normal_guided_approaches(
    local_normal,
    cone_angle_deg=15.0,
    num_azimuth=4,
):
    """Return the inward nominal approach and a single sparse cone around it."""
    normal = np.asarray(local_normal, dtype=float)
    length = np.linalg.norm(normal)
    if normal.shape != (3,) or not np.isfinite(length) or length <= 1e-12:
        raise ValueError("local_normal must be a non-zero finite 3-vector")
    cone_angle_deg = float(cone_angle_deg)
    if not np.isfinite(cone_angle_deg) or cone_angle_deg <= 0.0 or cone_angle_deg >= 90.0:
        raise ValueError("cone_angle_deg must be finite and in (0, 90)")
    if not isinstance(num_azimuth, int) or num_azimuth < 1:
        raise ValueError("num_azimuth must be a positive integer")

    nominal = -normal / length
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(reference @ nominal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    tangent_1 = np.cross(reference, nominal)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(nominal, tangent_1)
    tangent_2 /= np.linalg.norm(tangent_2)

    samples = [ApproachSample(approach_id=0, direction=nominal.copy(), offset_deg=0.0)]
    alpha = math.radians(cone_angle_deg)
    for azimuth_id in range(num_azimuth):
        phi = 2.0 * math.pi * azimuth_id / num_azimuth
        tangent = math.cos(phi) * tangent_1 + math.sin(phi) * tangent_2
        direction = math.cos(alpha) * nominal + math.sin(alpha) * tangent
        direction /= np.linalg.norm(direction)
        samples.append(
            ApproachSample(
                approach_id=azimuth_id + 1,
                direction=direction,
                offset_deg=cone_angle_deg,
            )
        )
    return samples
