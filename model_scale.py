"""Explicit input-unit scale configuration for the grasp pipeline."""

from __future__ import annotations

import os


DEFAULT_SCALE = 1000.0
MILLIMETRE_MODELS = {
    "1.ply", "2.ply", "2_mesh.ply", "huixing.ply", "juxing.ply",
    "mesh.ply", "sanjiao.ply", "shuilongtou.ply", "yuantai.ply",
    "yuanzhu.ply", "yuanzhu150.ply", "1_surface_output.ply",
}
METRE_DIRECTORIES = {"colmap", "0623"}


def get_model_scale(model_path: str | os.PathLike) -> float:
    """Return the multiplier converting input coordinates to internal mm."""
    path = os.path.normpath(os.fspath(model_path))
    parts = {part.lower() for part in path.replace("\\", "/").split("/")}
    filename = os.path.basename(path).lower()
    if parts.intersection(METRE_DIRECTORIES):
        return 1000.0
    if filename in MILLIMETRE_MODELS:
        return 1.0
    return DEFAULT_SCALE
