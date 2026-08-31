"""Deterministic random-state setup for complete annotation runs."""

from __future__ import annotations

import random

import numpy as np
import open3d as o3d


def configure_determinism(enabled=True, seed=0):
    """Seed every random source used by the current geometry/collision path."""
    if not enabled:
        return
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
