# Multi-view Grasp Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a view-conditioned, normal-filtered 6D grasp dataset pipeline while keeping the existing `grasp_detect(ply_path, i)` interface callable.

**Architecture:** `grasp_detect.py` gets a new surface-input adapter and reuses its current candidate, collision, and inner-OBB filters. New modules sample Fibonacci views, filter normal-facing surface points, aggregate and deduplicate candidates, and serialize training annotations.

**Tech Stack:** Python 3.10, NumPy, Open3D, SciPy, standard-library `unittest`.

---

### Task 1: Viewpoint sampling and front-facing filtering

**Files:**
- Create: `view_sampling.py`
- Create: `surface_visibility.py`
- Create: `tests/__init__.py`
- Create: `tests/test_view_sampling.py`
- Create: `tests/test_surface_visibility.py`

- [ ] **Step 1: Write failing unit tests for the public functions.**

```python
import unittest
import numpy as np
from view_sampling import generate_viewpoints
from surface_visibility import filter_front_facing_surface

class ViewTests(unittest.TestCase):
    def test_viewpoints_are_unit_vectors(self):
        views = generate_viewpoints(20)
        self.assertEqual(views.shape, (20, 3))
        np.testing.assert_allclose(np.linalg.norm(views, axis=1), 1.0)

    def test_normal_filter_removes_back_faces(self):
        points = np.array([[0, 0, 0], [1, 0, 0]], float)
        normals = np.array([[0, 0, 1], [0, 0, -1]], float)
        kept, _, mask = filter_front_facing_surface(points, normals, [0, 0, 1])
        np.testing.assert_array_equal(mask, [True, False])
        np.testing.assert_array_equal(kept, [[0, 0, 0]])
```

- [ ] **Step 2: Run the tests to confirm RED.**

Run: `F:\Miniconda\envs\py310\python.exe -m unittest tests.test_view_sampling tests.test_surface_visibility -v`

Expected: imports fail because both production modules are absent.

- [ ] **Step 3: Implement the minimum functions.**

```python
def generate_viewpoints(num_views=60):
    if not isinstance(num_views, int) or num_views <= 0:
        raise ValueError("num_views must be a positive integer")
    i = np.arange(num_views, dtype=float)
    z = 1 - 2 * (i + 0.5) / num_views
    radius = np.sqrt(1 - z * z)
    theta = np.pi * (3 - np.sqrt(5)) * i
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))

def filter_front_facing_surface(points, normals, view_direction, min_dot=1e-8):
    points, normals, view = (np.asarray(x, dtype=float) for x in (points, normals, view_direction))
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("points and normals must both have shape (N, 3)")
    if view.shape != (3,) or not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("inputs must be finite")
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    unit = np.zeros_like(normals)
    unit[valid] = normals[valid] / lengths[valid, None]
    mask = valid & (unit @ (view / np.linalg.norm(view)) > min_dot)
    return points[mask], unit[mask], mask
```

- [ ] **Step 4: Run the tests to confirm GREEN, then commit `feat: add virtual view and surface filtering primitives`.**

Run: `F:\Miniconda\envs\py310\python.exe -m unittest tests.test_view_sampling tests.test_surface_visibility -v`

Expected: all primitive tests pass.

### Task 2: Add the surface-conditioned detector API

**Files:**
- Modify: `grasp_detect.py:253-770`
- Create: `tests/test_surface_detector.py`

- [ ] **Step 1: Write failing tests for input validation and legacy compatibility.**

```python
import inspect
import unittest
import numpy as np
from grasp_detect import grasp_detect, grasp_detect_from_surface

class SurfaceDetectorTests(unittest.TestCase):
    def test_empty_surface_returns_empty_list(self):
        self.assertEqual(grasp_detect_from_surface(np.empty((0, 3)), np.empty((0, 3)), [0, 0, 1]), [])

    def test_nan_surface_is_rejected(self):
        with self.assertRaises(ValueError):
            grasp_detect_from_surface(np.array([[np.nan, 0, 0]]), np.array([[0, 0, 1]]), [0, 0, 1])

    def test_legacy_signature_is_preserved(self):
        self.assertEqual(list(inspect.signature(grasp_detect).parameters), ["ply_path", "i"])
```

- [ ] **Step 2: Run the test to confirm RED.**

Run: `F:\Miniconda\envs\py310\python.exe -m unittest tests.test_surface_detector -v`

Expected: import failure for missing `grasp_detect_from_surface`.

- [ ] **Step 3: Refactor only the reusable legacy candidate block.**

Extract the post-contact candidate generation from `grasp_detect` into `_generate_candidates_from_frame(cloud_down, frame, T_object_world, center0)`. Keep the existing gripper parameters, collision filter and inner-OBB filter. At its first line add:

```python
if center0 is None or not np.isfinite(center0).all():
    return [], []
```

Before `idx[0]` in `check_collision`, add:

```python
sampled_points = np.asarray(gripper_pcd.points)
if len(sampled_points) == 0 or not np.isfinite(sampled_points).all() or k == 0:
    return True
```

- [ ] **Step 4: Implement the new public adapter.**

```python
def grasp_detect_from_surface(surface_points, surface_normals, view_direction, metadata=None):
    points, normals, view = (np.asarray(x, dtype=float) for x in (surface_points, surface_normals, view_direction))
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("surface points and normals must have shape (N, 3)")
    if view.shape != (3,) or not np.isfinite(points).all() or not np.isfinite(normals).all() or not np.isfinite(view).all():
        raise ValueError("surface input must be finite")
    if len(points) == 0:
        return []
    # Build an Open3D cloud and orthonormal view frame, locate the first contact,
    # delegate to _generate_candidates_from_frame, attach metadata and view_direction.
```

The new function returns `list[dict]`; each record includes `T_gripper_object`, `opening`, `view_direction`, and provenance metadata. The old `grasp_detect(ply_path, i)` signature and five-value return tuple remain unchanged.

- [ ] **Step 5: Run the tests to confirm GREEN, then commit `feat: add surface-conditioned grasp detector`.**

Run: `F:\Miniconda\envs\py310\python.exe -m unittest tests.test_surface_detector -v`

Expected: validation, empty-input, and old-signature tests pass.

### Task 3: Aggregate views, score candidates, and remove duplicates

**Files:**
- Create: `grasp_merge.py`
- Create: `multi_view_grasp.py`
- Create: `tests/test_grasp_merge.py`
- Create: `tests/test_multi_view_grasp.py`

- [ ] **Step 1: Write failing merge and dependency-injected orchestration tests.**

```python
def test_keeps_higher_scored_duplicate():
    first = {"T_gripper_object": np.eye(4), "score_total": 0.2}
    second = {"T_gripper_object": np.eye(4), "score_total": 0.8}
    assert merge_duplicate_grasps([first, second], 5.0, 10.0) == [second]

def test_calls_detector_once_per_view():
    calls = []
    def detector(points, normals, view, metadata):
        calls.append(metadata["view_id"])
        return [{"T_gripper_object": np.eye(4), "opening": 50.0}]
    result = generate_multi_view_grasps("ignored", 3, loader=lambda _: (np.zeros((3, 3)), np.tile([0, 0, 1.0], (3, 1))), detector=detector, scorer=lambda grasps, _: grasps, deduplicate=False)
    assert calls == [0, 1, 2]
```

- [ ] **Step 2: Run the tests to confirm RED.**

Run: `F:\Miniconda\envs\py310\python.exe -m unittest tests.test_grasp_merge tests.test_multi_view_grasp -v`

Expected: missing production-module imports.

- [ ] **Step 3: Implement pose merging.**

```python
def merge_duplicate_grasps(grasps, position_threshold_mm=5.0, rotation_threshold_deg=10.0):
    selected = []
    for grasp in sorted(grasps, key=lambda g: g.get("score_total", float("-inf")), reverse=True):
        pose = np.asarray(grasp["T_gripper_object"], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            continue
        if not any(_is_duplicate(pose, np.asarray(item["T_gripper_object"]), position_threshold_mm, rotation_threshold_deg) for item in selected):
            selected.append(grasp)
    return selected
```

`_is_duplicate` compares translation Euclidean distance and the rotation angle from `R1.T @ R2`.

- [ ] **Step 4: Implement the orchestrator.**

```python
@dataclass
class MultiViewResult:
    grasps: list[dict]
    skipped_views: list[dict]
    view_candidate_counts: dict[int, int]

def generate_multi_view_grasps(path, num_views=60, position_threshold_mm=5.0, rotation_threshold_deg=10.0, *, loader=_load_cloud, detector=grasp_detect_from_surface, scorer=compute_grasp_scores_simple, deduplicate=True):
    points, normals = loader(path)
    for view_id, view in enumerate(generate_viewpoints(num_views)):
        visible_points, visible_normals, _ = filter_front_facing_surface(points, normals, view)
        # Record empty views, call detector, score candidates, attach view_id and score_total.
    # Merge only after all views, then return MultiViewResult.
```

The default loader reads the PLY once and estimates normals only when missing. `_score_total` prefers finite `score_force_closure`, then finite `score_inner_points_ratio`; all legacy score fields stay in the record.

- [ ] **Step 5: Run tests to confirm GREEN, then commit `feat: aggregate and deduplicate multi-view grasps`.**

Run: `F:\Miniconda\envs\py310\python.exe -m unittest tests.test_grasp_merge tests.test_multi_view_grasp -v`

Expected: merge and per-view orchestration tests pass.

### Task 4: Save the training dataset and add the py310 command

**Files:**
- Create: `grasp_database.py`
- Create: `main.py`
- Create: `tests/test_grasp_database.py`

- [ ] **Step 1: Write a failing output test.**

```python
result = save_grasp_dataset([{"T_gripper_object": np.eye(4), "opening": 50.0, "score_total": 0.7, "view_id": 1}], tmp_path, {"units": "mm"})
assert result.json_path.exists() and result.npz_path.exists() and result.meta_path.exists()
assert np.load(result.npz_path)["poses"].shape == (1, 4, 4)
```

- [ ] **Step 2: Run the test to confirm RED.**

Run: `F:\Miniconda\envs\py310\python.exe -m unittest tests.test_grasp_database -v`

Expected: missing `grasp_database` module.

- [ ] **Step 3: Implement dataset serialization.**

`save_grasp_dataset` creates `grasps.json`, `grasps.npz`, and `meta.json`. Each JSON record stores ID, translation, flattened rotation, SciPy `xyzw` quaternion, opening, `score_total`, scalar `score_*` fields, `view_id`, and view direction. NPZ keys are `poses`, `translations`, `rotations`, `quaternions`, `openings`, `scores`, and `view_ids`.

- [ ] **Step 4: Implement the command line.**

```python
parser.add_argument("--object", required=True)
parser.add_argument("--views", type=int, default=60, choices=[20, 40, 60, 100])
parser.add_argument("--output", required=True)
parser.add_argument("--position-threshold-mm", type=float, default=5.0)
parser.add_argument("--rotation-threshold-deg", type=float, default=10.0)
```

The command prints processed/skipped views, per-view candidate counts, raw/deduplicated counts, and output locations.

- [ ] **Step 5: Run the test to confirm GREEN, then commit `feat: save multi-view grasp annotations`.**

Run: `F:\Miniconda\envs\py310\python.exe -m unittest tests.test_grasp_database -v`

Expected: JSON, NPZ, and metadata files exist with the expected pose shape.

### Task 5: Regression and smoke verification

**Files:**
- No planned source changes.

- [ ] **Step 1: Run all tests with py310.**

Run: `F:\Miniconda\envs\py310\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 2: Run a bounded real-cloud test.**

Run: `F:\Miniconda\envs\py310\python.exe main.py --object model/huixing.ply --views 20 --output results/huixing-smoke`

Expected: no `IndexError`; `grasps.json`, `grasps.npz`, and `meta.json` exist. Never stage generated `results/` or the existing untracked `.vscode/` directory.
