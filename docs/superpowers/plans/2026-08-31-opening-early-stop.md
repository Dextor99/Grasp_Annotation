# Opening First-Free Early-Stop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax to track progress.

**Goal:** Select the minimum collision-free opening for each `(depth, angle)` grasp group while preserving legacy collision and downstream filtering behavior.

**Architecture:** Keep the generic `filter_collision_free_grippers` API unchanged. Add a group-aware filter that orders each `(depth, angle_deg)` group by opening and stops after the first collision-free candidate; `grasp_detect` uses this filter after structural pre-filtering, while the existing minimum-opening filter remains as a safety check.

**Tech Stack:** Python 3.10, NumPy, Open3D, unittest.

---

### Task 1: Define group-aware first-free behavior

**Files:**
- Create: `tests/test_opening_early_stop.py`

- [x] Test that openings are evaluated in ascending order per `(depth, angle_deg)`, later openings in a group are skipped after the first collision-free candidate, and groups with no free opening are fully evaluated.
- [x] Run the focused test and verify the expected missing-function failure.

### Task 2: Implement and integrate early-stop

**Files:**
- Modify: `grasp_detect.py`
- Modify: `tests/test_opening_early_stop.py`

- [x] Add `filter_collision_free_grippers_first_opening` without changing the generic collision filter.
- [x] Call it from `grasp_detect` after `filter_structurally_valid_grippers`; keep `filter_by_min_opening_per_depth_angle` downstream as a safety guard and preserve profiling fields.
- [x] Run focused and complete tests.

### Task 3: Remove unused anchor cylinder search

**Files:**
- Modify: `grasp_detect.py`
- Modify: `tests/test_surface_api.py`

- [x] Add a regression assertion that anchor calls do not invoke `generate_cylinder_sections`.
- [x] Skip cylinder generation when `contact_center_override` is supplied, set `center0=contact_center_override`, `center1=center0 + z_axis*40`, and keep legacy calls unchanged.
- [x] Run focused and complete tests plus a real one-view cone smoke.

### Task 4: Review, commit, and push

**Files:**
- No additional files.

- [x] Run full tests and `git diff --check` with Python 3.10.
- [x] Commit only early-stop/cylinder-cleanup files and tests, preserving user-local files, then push `main`.
