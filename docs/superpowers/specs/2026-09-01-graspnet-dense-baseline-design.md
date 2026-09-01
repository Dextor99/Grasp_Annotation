# GraspNet-1Billion-Style Dense Annotation Baseline

## Status and purpose

This document specifies an independent baseline that reproduces the *object-level dense grasp annotation protocol* used by GraspNet-1Billion.  It is **not** a reproduction of the GraspNet neural detector.  The baseline will be used to compare dense, exhaustive approach sampling with the frozen `v1.2-grasp-annotation` method.

The frozen project pipeline, its V4 scorer, closure refinement, merge logic, and formal experiment results must not be modified by this work.

## Evidence and compatibility target

The official raw label representation stores:

- `points`: sampled grasp points in model coordinates;
- `offsets`: per-point `(view, in-plane angle, depth, width)` values;
- `collision`: collision mask;
- `scores`: minimum coefficient of friction for stable grasping, where lower is better and `-1` denotes invalid.

The dense candidate topology is fixed at 300 viewpoints, 12 in-plane rotations, and 4 depths.  Therefore, a fully expanded grasp point has exactly `300 * 12 * 4 = 14,400` candidates.  The baseline preserves this topology and score meaning.

Sources:

- Official GraspNet API label specification: <https://graspnetapi.readthedocs.io/en/latest/grasp_format.html>
- Official API: <https://github.com/graspnet/graspnetAPI>
- Public auto-annotation reference (parameter/reference aid only, not an official source): <https://github.com/rhett-chen/grasp-auto-annotation>

## Scope

### In scope

1. An isolated `baselines/graspnet_annotation/` package.
2. Explicit conversion of every input mesh/point cloud into metres, with provenance saved alongside every output.
3. Dense candidate generation with official 300-view sampling, 12 angles, and 4 approach depths.
4. Opening/width estimation, official-compatible gripper geometry checks, collision masking, and Dex-Net force-closure evaluation.
5. Export of raw-format-compatible arrays, a compact valid-grasp representation, per-stage timing, configuration, seed, and source-parameter provenance.
6. A formal `GN-Full` mode and a later `GN-Fair/Budget` mode that only changes candidate budget, not evaluation semantics.
7. Test-first validation at 18 candidates, 14,400 candidates, 288,000 candidates, then streamed full generation.

### Out of scope

- Any modification of the frozen ours pipeline.
- Neural-network training, inference, or claims of reproducing the GraspNet detector.
- V4 scoring during raw GraspNet-style annotation.
- A comparison table before both methods have been evaluated against the same reference mesh and force-closure protocol.

## Geometry, units, and dependencies

Inputs are converted once at the package boundary and are thereafter always expressed in metres.  No unit-sphere normalization is allowed because GraspNet-style widths, depths, collision dimensions, and SDF values have physical units.

The first full validation asset is `model/2.stl`.  It must pass mesh loading and watertightness diagnostics.  A signed-distance field is required for Dex-Net quality evaluation; non-watertight geometry is a blocking data-preparation error, not a reason to silently change the algorithm.

The implementation will prefer the official `graspnetAPI` and its compatible Dex-Net dependencies for pose conversion, collision semantics, and force-closure evaluation.  Dependency availability is checked explicitly before any complete run.  A missing dependency produces an actionable setup error and does not yield partial results labelled as force-closure scored.

## Package boundary

```text
baselines/graspnet_annotation/
  __init__.py
  config.py                  # immutable run configuration and official/default provenance
  preprocess.py              # mesh loading, metres conversion, mesh/SDF readiness checks
  grasp_point_sampling.py    # seeded surface/voxel sampling with caps
  view_sampling.py           # official Fibonacci view ordering
  candidate_generation.py    # streamed point/view/angle/depth expansion
  width_estimation.py        # width offsets from local object geometry
  official_adapter.py        # graspnetAPI/Dex-Net conversion, collision, force closure
  export.py                  # raw arrays, valid grasp array, summary and timing
  run_graspnet_baseline.py   # single documented CLI entry point
  tests/
```

No module in this package imports or mutates `multi_view_grasp.py`, `grasp_pipeline.py`, `grasp_score_v4.py`, or `grasp_merge.py`.

## Frozen GN-Full protocol

| Parameter | Value | Status |
| --- | ---: | --- |
| surface samples | 6000 | public reimplementation-derived default |
| voxel sampling | 0.006 m | public reimplementation-derived default |
| point cap | 1200 | public reimplementation-derived default |
| random seed | 0 | baseline configuration |
| viewpoints | 300 | official label topology |
| in-plane angles | 12, from 0 to pi in 15 degree increments | official label topology |
| depths | 0.01, 0.02, 0.03, 0.04 m | official label topology |
| gripper height | 0.02 m | public reference default |
| gripper depth base | 0.02 m | public reference default |
| finger width | 0.01 m | public reference default |
| maximum opening | 0.12 m | public reference default |
| empty threshold | 10 points | public reference default |
| loose collision margin | 0.004 m | public reference default |
| friction sweep | 1.0 down to 0.1 in steps of 0.1 | baseline evaluation configuration |

Each exported field records whether it is official topology, public-reference default, or local baseline configuration.  This avoids presenting non-official tuning values as official GraspNet parameters.

## Processing flow

```text
STL / OBJ / PLY
  -> explicit metres conversion and mesh/SDF readiness validation
  -> deterministic grasp-point sampling
  -> 300 Fibonacci approach views
  -> 12 in-plane angles x 4 depths, generated in batches
  -> width estimation
  -> official-compatible collision and empty-space mask
  -> Dex-Net force-closure sweep
  -> raw-format export and compact valid-grasp export
```

Candidate tensors are streamed point batches rather than materialising all `N * 14,400` poses in memory.  The export remains shape-compatible with the official raw-label topology.

## Outputs

Each completed run writes one output directory containing:

- `grasp_labels.npz`: `points`, `offsets`, `collision`, `scores` with official-style axes;
- `valid_grasps.npy`: compact, documented pose representation for valid candidates;
- `summary.json`: counts, valid rate, width/depth distributions, score distribution, units, seed, configuration, and parameter provenance;
- `timing.csv`: preprocess, sampling, candidate generation, collision, force closure, export, and total timings;
- `run_config.json`: exact reproducibility record.

`scores` remain minimum friction coefficients.  A derived quality score such as `1.1 - mu_min` may be placed in `summary.json` for plotting, but never replaces raw `scores`.

## CLI

```powershell
F:\Miniconda\envs\py310\python.exe baselines\graspnet_annotation\run_graspnet_baseline.py `
  --mesh model\2.stl `
  --input-unit mm `
  --mode full `
  --output results\baselines\graspnet\2_full
```

The command rejects unknown units, missing SDF/Dex-Net prerequisites, and mode/configuration combinations that would create a misleading partial label set.

## Acceptance tests

1. **Dependency/readiness gate:** precise diagnostic for unavailable official API, Dex-Net, or mesh/SDF prerequisites.
2. **Debug topology:** one point, three views, three angles, two depths yields 18 candidates.
3. **Official topology:** one point, 300 views, 12 angles, four depths yields 14,400 candidates with expected tensor axes.
4. **Streaming test:** 20 points produces 288,000 candidates without full-pose materialisation.
5. **Determinism:** same mesh/config/seed produces byte-equivalent discrete labels and numerically equal continuous output within documented tolerance.
6. **Unit test:** a known 100 mm feature becomes 0.1 m internally; depths and openings remain in metres.
7. **Integration validation:** `model/2.stl` completes all stages with finite valid scores or a clear geometry/SDF error.

## Follow-on GN-Fair/Budget protocol

After GN-Full passes the acceptance suite, GN-Fair/Budget will reduce only the number of sampled grasp points and/or candidate batches.  It will retain the same unit conversion, gripper, collision, force-closure, reference mesh, and raw score semantics.  Cross-method reporting will use common reference geometry and force-closure evaluation; this comparison is deliberately separate from the reproduction implementation.
