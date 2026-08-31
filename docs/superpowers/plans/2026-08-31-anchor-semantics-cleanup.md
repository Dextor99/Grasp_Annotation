# Anchor Semantics Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the first surface anchor central-representative and make the anchor grasp adapter's pose semantics explicitly use the anchor as the depth-zero contact reference.

**Architecture:** Keep the existing global/normal/cone pipeline and collision core unchanged. Only change FPS seeding and the new anchor adapter frame/contact center; legacy `grasp_detect(ply_path, i)` retains its original spherical contact search.

**Tech Stack:** Python 3.10, NumPy, Open3D, unittest.

---

### Task 1: Centroid-seeded FPS

**Files:**
- Modify: `surface_anchor.py`
- Modify: `tests/test_surface_anchor.py`

- [x] Assert that the first selected index is the point nearest the visible-surface centroid and that duplicate indices are still avoided.
- [x] Run the focused test and observe the expected failure against the current farthest-from-centroid seed.
- [x] Change only the FPS seed from `argmax` distance to `argmin` distance; preserve deterministic farthest-point iterations and distance threshold.
- [x] Run focused and complete tests.

### Task 2: Explicit anchor depth-zero adapter

**Files:**
- Modify: `grasp_detect.py`
- Modify: `multi_view_grasp.py`
- Modify: `tests/test_surface_api.py`
- Modify: `tests/test_multi_view_grasp.py`

- [x] Update the adapter test to require `frame.origin == anchor`, `frame.z_axis == approach`, and `contact_center_override == anchor` without an `approach_offset_mm` argument.
- [x] Set the anchor adapter frame origin and contact center directly to the anchor; remove the unused approach-offset parameter from this path while leaving legacy APIs untouched.
- [x] Update multi-view calls and tests accordingly; preserve metadata and all full-cloud filtering.
- [x] Run focused and complete tests.

### Task 3: Cross-model validation and push

**Files:**
- No additional production files.

- [x] Preprocess `model/colmap/cat.ply`, `model/huixing.ply`, and `model/shuilongtou.ply`; report normal component standard deviations and nonzero front-facing counts for five views.
- [x] Run a real `cat.ply` 1-view × 2-anchor × 5-approach smoke with visualization disabled.
- [x] Commit only this plan, anchor/FPS/adapter/multi-view changes and tests; preserve user-local files, then push `main`.
