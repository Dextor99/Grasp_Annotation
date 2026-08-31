# Hierarchical Surface Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace one centroid approach per global view with FPS surface anchors and sparse local-normal-guided approach cones while preserving the existing collision/filter core.

**Architecture:** Global Fibonacci views continue to select front-facing surface subsets. Each subset produces deterministic FPS anchors with smoothed local normals; each anchor produces one nominal inward approach plus four cone approaches. A new anchor/approach adapter builds the local frame and delegates to the existing `grasp_detect` core, which continues to validate against the full object cloud.

**Tech Stack:** Python 3.10, NumPy, scikit-learn KDTree, Open3D, unittest.

---

### Task 1: Surface anchors

**Files:**
- Create: `surface_anchor.py`
- Create: `tests/test_surface_anchor.py`

- [x] Write tests proving FPS is deterministic, spatially separated, and local normals are normalized and flipped toward `view_direction`.
- [x] Run `python -m unittest tests.test_surface_anchor -v` and verify it fails because `surface_anchor` does not exist.
- [x] Implement `SurfaceAnchor`, `farthest_point_sample_indices`, `estimate_local_normal`, and `build_surface_anchors` with `normal_knn=30`.
- [x] Run the focused test and the complete test suite.

### Task 2: Normal-guided approach cone

**Files:**
- Create: `approach_sampling.py`
- Create: `tests/test_approach_sampling.py`

- [x] Write tests proving the result contains one `-normal` approach and four unit vectors exactly 15 degrees from it.
- [x] Run `python -m unittest tests.test_approach_sampling -v` and verify the missing-module failure.
- [x] Implement `ApproachSample` and `sample_normal_guided_approaches(local_normal, cone_angle_deg=15, num_azimuth=4)`.
- [x] Run focused and complete tests.

### Task 3: Anchor/approach grasp adapter

**Files:**
- Modify: `grasp_detect.py`
- Modify: `tests/test_surface_api.py`

- [x] Add a failing test for `grasp_detect_from_anchor_approach` that asserts `frame.origin = anchor_point - approach * offset` and `frame.z_axis = approach`.
- [x] Implement frame construction with a stable perpendicular reference axis and delegate to the existing `grasp_detect(..., object_data=..., frame_override=...)` path.
- [x] Attach anchor and approach metadata without changing collision, depth, opening, or filtering logic.
- [x] Run focused and complete tests.

### Task 4: Three-level multi-view loop

**Files:**
- Modify: `multi_view_grasp.py`
- Create: `tests/test_multi_view_grasp.py`

- [x] Add failing tests for modes `global`, `normal`, and `cone`, metadata fields, and one-time object preprocessing.
- [x] Extend `generate_multi_view_grasps` with `mode`, `num_anchors_per_view`, `cone_angle_deg`, `num_approach_azimuth`, and `normal_knn`.
- [x] Implement `view -> surface -> anchor -> approach` loops; keep the current centroid/global path as Variant A.
- [x] Run `1 view x 2 anchors x 5 approaches` against `model/colmap/cat.ply`, with visualization disabled, and verify all collision checks still use `object_data.cloud_down`.
- [x] Run the full test suite and commit only task files, preserving local user changes.
