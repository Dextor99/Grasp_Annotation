# Multi-view Grasp Dataset Design

## Goal

Extend the project with a view-conditioned, normal-filtered 6D grasp dataset generator while preserving the existing `grasp_detect(ply_path, i)` workflow.

## Scope and compatibility

- Keep the legacy `grasp_detect`, `cloud_process`, scoring, gripper, and collision workflows available.
- Add a new public `grasp_detect_from_surface(surface_points, surface_normals, view_direction, metadata=None)` API in `grasp_detect.py`.
- Add new orchestration, visibility, persistence, and command-line modules.
- Run all validation with `F:\Miniconda\envs\py310\python.exe`.
- Do not alter existing models, score records, or legacy executable scripts.

## Architecture

```text
PLY / future COLMAP point cloud
  -> load points and normals once
  -> Fibonacci virtual viewpoints
  -> normal-based front-facing surface filtering
  -> grasp_detect_from_surface
  -> existing candidate, collision, and internal-point filters
  -> grasp_score_V3 evaluation
  -> pose deduplication and ranking
  -> JSON + NPZ annotation dataset
```

`multi_view_grasp.py` owns scheduling. It creates a Fibonacci direction, filters the input cloud for that direction, passes surface data and provenance into the new detector API, scores candidates, then aggregates all successful views. A view producing no visible points or no valid candidates is recorded and skipped rather than failing the whole run.

## Detector interface

The existing function remains callable without changes:

```python
grasp_detect("model/huixing.ply", 194)
```

The new API accepts already-prepared view-conditioned data:

```python
grasp_detect_from_surface(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    view_direction: np.ndarray,
    metadata: dict | None = None,
) -> list[dict]
```

It validates finite `(N, 3)` points and normals, constructs a local grasp frame from the view direction and surface support, and reuses the established gripper generation, collision filtering, and inner-volume filtering routines. Every resulting grasp receives `view_id`, `view_direction`, and the supplied metadata. Invalid or non-intersecting input returns an empty candidate list, never a pose containing `NaN`.

## Visibility semantics

The first version performs **normal-based front-facing filtering**, not ray-cast visibility. With a view vector directed from the object toward the virtual camera, a point is retained when its normal faces the camera (using one documented dot-product convention). This removes back-facing regions but does not guarantee removal of self-occluded surfaces. The module must state this limitation in its public documentation so the paper does not overclaim visibility extraction.

## Scoring, merging, and output

`grasp_score_V3.compute_grasp_scores_simple` adds the existing score components. The pipeline preserves all available numeric score fields and selects a deterministic `score_total` for ranking: `score_force_closure` when finite, falling back to `score_inner_points_ratio`. It must not mutate the legacy scorer.

Near-duplicate poses are merged after all views. Two poses are duplicates when their translations are within a configurable threshold and their rotations differ by no more than a configurable angle. The higher-ranked pose is kept.

For each retained grasp, JSON stores:

- stable integer `id`;
- translation and 3x3 rotation flattened row-major;
- quaternion in `xyzw` order;
- gripper opening/width;
- `score_total` and the individual numeric score fields;
- collision status when available;
- `view_id`, `view_direction`, and source metadata.

NPZ stores dense arrays for pose, width, score, and view IDs for training. A metadata JSON file records units, the visibility strategy, gripper settings, input path, and run parameters.

## Command line

`main.py` provides:

```powershell
F:\Miniconda\envs\py310\python.exe main.py --object model/huixing.ply --views 60 --output results/huixing
```

The command writes `grasps.json`, `grasps.npz`, and `meta.json` under the output directory. It reports processed, skipped, raw, deduplicated, and saved grasp counts.

## Error handling

- Empty, malformed, non-finite, or mismatched surface arrays raise clear `ValueError`s at the public boundary.
- A non-intersecting view returns an empty set with a skip reason; it does not enter collision detection.
- Empty meshes, empty KD-tree results, and non-finite sampled coordinates are rejected by collision-check guards.
- Per-view failures are collected in run metadata and do not discard successful views.

## Tests and verification

New tests will cover Fibonacci direction normalization/distribution, front-facing normal filtering, detector input validation and no-`NaN` behavior, pose merge rules, JSON/NPZ serialization, and multi-view aggregation using a lightweight injected detector. Tests for the unchanged legacy detector are limited to an import-and-signature regression check because executing it is computationally expensive and model-dependent.

All tests run with the py310 interpreter. A small local PLY smoke run will verify that the new command emits readable annotations without modifying legacy output files.

## Deferred scope

- True mesh ray-casting and self-occlusion handling are deferred until a watertight COLMAP mesh is available.
- Changing the existing scoring model or its collision model is out of scope.
- Visualization output is deferred; the existing Open3D visualization remains usable for inspection.
