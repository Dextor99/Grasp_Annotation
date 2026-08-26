"""Reusable, one-time point-cloud preprocessing for view-conditioned grasping."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.neighbors import KDTree

from cloud_process import frames_process
from model_scale import get_model_scale


@dataclass
class ObjectData:
    """Preprocessed object state shared by all view-level grasp calls."""

    ply_path: str
    cloud_down: object
    points: np.ndarray
    normals: np.ndarray
    kdtree: KDTree
    center: np.ndarray
    radius: float
    scale: float
    obj_axes: np.ndarray
    sample_points: np.ndarray
    frames: list
    object_world_axis: tuple
    projections: list
    frame_arrows_list: list
    T_object_world: np.ndarray

    @property
    def sample_radius(self) -> float:
        return float(np.linalg.norm(self.sample_points[0] - self.center))

    @property
    def frames_result(self) -> tuple:
        """Return the legacy frames_process tuple for core reuse."""
        return (
            self.cloud_down,
            self.center,
            self.obj_axes,
            self.sample_points,
            self.frames,
            self.object_world_axis,
            self.projections,
            self.frame_arrows_list,
            self.T_object_world,
        )


def prepare_object(ply_path: str) -> ObjectData:
    """Load, scale, downsample and estimate normals exactly once."""
    result = frames_process(ply_path)
    cloud_down, center, obj_axes, sample_points, frames, object_world_axis, projections, frame_arrows, transform = result
    points = np.asarray(cloud_down.points)
    normals = np.asarray(cloud_down.normals)
    sample_radius = float(np.linalg.norm(sample_points[0] - center))
    radius = sample_radius - 100.0
    if radius <= 0 or not np.isfinite(radius):
        radius = float(np.max(np.linalg.norm(points - center, axis=1)))
    return ObjectData(
        ply_path=str(ply_path),
        cloud_down=cloud_down,
        points=points,
        normals=normals,
        kdtree=KDTree(points),
        center=np.asarray(center),
        radius=radius,
        scale=get_model_scale(ply_path),
        obj_axes=np.asarray(obj_axes),
        sample_points=np.asarray(sample_points),
        frames=frames,
        object_world_axis=object_world_axis,
        projections=projections,
        frame_arrows_list=frame_arrows,
        T_object_world=np.asarray(transform),
    )
