"""Frozen configuration for the finalized grasp annotation method."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real


# Frozen evaluation-protocol values.  These are deliberately kept outside the
# generation search parameters: the threshold labels a finalized V4 grasp as
# high quality, but never changes candidate generation or scoring.
V4_HIGH_QUALITY_THRESHOLD = 0.13
V4_THRESHOLD_SOURCE = "manual_calibration"
V4_CALIBRATION_SAMPLES = 60
V4_CALIBRATION_GOOD = 19
V4_CALIBRATION_BAD = 25
V4_CALIBRATION_UNCERTAIN = 16


@dataclass(frozen=True)
class GraspGenerationConfig:
    num_views: int = 5
    anchors_per_view: int = 3
    mode: str = "cone"
    visibility_threshold: float = 0.0
    normal_knn: int = 30
    cone_angle_deg: float = 15.0
    num_approach_azimuth: int = 4
    depth_samples: int = 16
    depth_max_ratio: float = 1.2
    rotation_step_deg: int = 15
    rotation_max_deg: int = 179
    opening_step_mm: int = 15
    opening_max_mm: int = 150
    collision_threshold_mm: float = 3.0
    min_intersection_points: int = 5
    translation_merge_mm: float = 5.0
    rotation_merge_deg: float = 10.0
    closure_margin_mm: float = 2.0
    deterministic: bool = True
    random_seed: int = 0
    enable_visualization: bool = False

    def __post_init__(self):
        if self.mode not in {"global", "normal", "cone"}:
            raise ValueError("mode must be 'global', 'normal', or 'cone'")
        for name in (
            "num_views", "anchors_per_view", "normal_knn", "depth_samples",
            "rotation_step_deg", "opening_step_mm", "min_intersection_points",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.num_approach_azimuth) is not int or self.num_approach_azimuth < 1:
            raise ValueError("num_approach_azimuth must be positive")
        for name in (
            "cone_angle_deg", "depth_max_ratio", "collision_threshold_mm",
            "translation_merge_mm", "rotation_merge_deg",
            "closure_margin_mm",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("rotation_max_deg", "opening_max_mm"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")

    @property
    def num_approach_directions(self):
        return 1 if self.mode != "cone" else self.num_approach_azimuth + 1

    def to_dict(self):
        return {key: _json_primitive(value) for key, value in asdict(self).items()}


def _json_primitive(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    raise TypeError(f"unsupported configuration value: {type(value).__name__}")
