"""Configuration for the isolated GN-Full annotation baseline.

The raw label topology follows GraspNet's 300 viewpoints, 12 in-plane
angles, and 4 approach depths.  All physical lengths inside this package are
metres; input conversion is handled explicitly by :mod:`preprocess`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


_FRICTION_COEFFICIENTS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
_VALID_UNITS = ("m", "mm", "cm")


@dataclass(frozen=True)
class DenseAnnotationConfig:
    """Validated, reproducible configuration for GN-Full."""

    input_unit: str = "m"
    seed: int = 0
    surface_samples: int = 6000
    voxel_size_m: float = 0.006
    max_grasp_points: int = 1200
    num_views: int = 300
    num_angles: int = 12
    depths_m: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04)
    height_m: float = 0.02
    depth_base_m: float = 0.02
    finger_width_m: float = 0.01
    bottom_thickness_m: float = 0.1
    max_width_m: float = 0.12
    hole_size_m: float = 0.018
    width_loose_factor_m: float = 0.004
    empty_thresh: int = 10
    collision_margin_m: float = 0.004
    friction_coefficients: tuple[float, ...] = _FRICTION_COEFFICIENTS

    def __post_init__(self) -> None:
        if self.input_unit not in _VALID_UNITS:
            raise ValueError(f"input_unit must be one of {_VALID_UNITS}, got {self.input_unit!r}")
        for name in ("surface_samples", "max_grasp_points", "num_views", "num_angles", "empty_thresh"):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.depths_m or any(float(depth) <= 0 for depth in self.depths_m):
            raise ValueError("depths_m must contain positive values")
        for name in (
            "voxel_size_m", "height_m", "depth_base_m", "finger_width_m", "bottom_thickness_m",
            "max_width_m", "hole_size_m", "width_loose_factor_m", "collision_margin_m",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if any(not 0.0 < float(mu) <= 1.0 for mu in self.friction_coefficients):
            raise ValueError("friction_coefficients must be in (0, 1]")

    @classmethod
    def full(cls, **overrides: Any) -> "DenseAnnotationConfig":
        """Return the frozen full protocol, optionally with test overrides."""

        if "depths_m" in overrides:
            overrides["depths_m"] = tuple(float(value) for value in overrides["depths_m"])
        if "friction_coefficients" in overrides:
            overrides["friction_coefficients"] = tuple(float(value) for value in overrides["friction_coefficients"])
        return replace(cls(), **overrides)

    @property
    def candidates_per_point(self) -> int:
        return self.num_views * self.num_angles * len(self.depths_m)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["depths_m"] = list(self.depths_m)
        values["friction_coefficients"] = list(self.friction_coefficients)
        values["candidates_per_point"] = self.candidates_per_point
        return values

    def parameter_provenance(self) -> dict[str, list[str]]:
        return {
            "official_topology": ["num_views", "num_angles", "depths_m"],
            "public_reference_default": [
                "surface_samples", "voxel_size_m", "max_grasp_points", "height_m",
                "depth_base_m", "finger_width_m", "bottom_thickness_m", "max_width_m",
                "hole_size_m", "width_loose_factor_m", "empty_thresh", "collision_margin_m",
            ],
            "local_baseline_config": ["input_unit", "seed", "friction_coefficients"],
        }
