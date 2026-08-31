# Scale-Aware Normal Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make preprocessing normal estimation use a neighborhood radius derived from each model's downsample voxel size, so surface visibility and local anchors receive geometrically meaningful normals across model scales.

**Architecture:** Keep the existing `estimate_normals` API and all grasp-generation/collision logic unchanged. Compute `normal_radius = normal_radius_factor * voxel_size` inside `frames_process`, pass it to the existing estimator, and expose a small pure helper for deterministic validation.

**Tech Stack:** Python 3.10, NumPy, Open3D, unittest.

---

### Task 1: Define scale-aware radius behavior with tests

**Files:**
- Modify: `tests/test_cloud_process.py`
- Modify: `docs/superpowers/plans/2026-08-31-scale-aware-normal-estimation.md`

- [ ] Add tests for the default factor (`2.5 * voxel_size`) and invalid voxel sizes/factors.
- [ ] Run `F:\Miniconda\envs\py310\python.exe -m unittest tests.test_cloud_process -v`; expected failure is the missing helper.

### Task 2: Use voxel-derived radius in preprocessing

**Files:**
- Modify: `cloud_process.py`
- Modify: `tests/test_cloud_process.py`

- [ ] Add `normal_search_radius(voxel_size, factor=2.5)` with finite-positive validation.
- [ ] In `frames_process`, compute and print the radius from the already computed voxel size, then pass it to `estimate_normals` while preserving `max_nn=30` and all downstream behavior.
- [ ] Run the focused tests and the complete suite; all tests must pass.

### Task 3: Cross-model normal and visibility smoke validation

**Files:**
- No production files beyond Task 2.

- [ ] Run preprocessing for `model/colmap/cat.ply`, `model/huixing.ply`, and `model/shuilongtou.ply` and print normal component standard deviations plus front-facing counts for five Fibonacci views.
- [ ] Verify the command exits successfully and no model produces an estimator exception; record the observed values before deciding any later optimization.
- [ ] Commit only scale-aware normal files/tests/plan, preserving existing user changes, then push `main`.
