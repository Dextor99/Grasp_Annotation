# Finalize Grasp Annotation Generator Implementation Plan

> **For Codex:** Execute this plan task-by-task with TDD. Keep legacy entry points compatible, preserve the user's local `grasp_score_calculate.py` and `.vscode/` changes, and commit each feature phase separately.

**Goal:** Freeze the current geometry method behind a reproducible pipeline that scores, symmetry-merges, normalizes, and exports final 6-DoF grasp annotations.

**Architecture:** Keep `generate_multi_view_grasps()` as the raw geometry generator. Add small composable modules for scoring, SE(3) merging, deterministic setup, schema normalization, orchestration, and export. The new `main.py` is the only end-user entry point and calls the modules in the fixed order `prepare -> generate -> score -> merge -> export`.

**Tech Stack:** Python 3.10, NumPy, SciPy, Open3D, unittest, JSON, NumPy NPZ.

---

## Task 1: Score and rank multi-view candidates

**Files:**
- Create: `grasp_scoring.py`
- Modify: `multi_view_grasp.py`
- Test: `tests/test_grasp_scoring.py`

1. Write failing tests for finite `score_total`, force-closure preference with inner-ratio fallback, descending order, and preservation of existing score components.
2. Run `F:\Miniconda\envs\py310\python.exe -m unittest tests.test_grasp_scoring -v` and confirm the missing module/API failure.
3. Implement `score_grasp_candidates(object_data, grasps)` as a wrapper over `compute_grasp_scores_simple`; do not change the legacy score equations.
4. Add a scored multi-view adapter without changing the raw generator's return contract.
5. Run the focused test and the existing multi-view tests.
6. Commit as `feat: score and rank multi-view grasp candidates`.

## Task 2: Merge redundant poses in symmetry-aware SE(3)

**Files:**
- Create: `grasp_merge.py`
- Test: `tests/test_grasp_merge.py`

1. Write failing tests for translation/rotation thresholds, 180-degree parallel-jaw symmetry, best-score retention, provenance union, and stable score order.
2. Run the focused test and confirm failure.
3. Implement greedy score-ordered merge with defaults `translation < 5 mm`, `rotation < 10 deg`; compare both the normal relative rotation and a 180-degree rotation around the local gripper approach/symmetry axis.
4. Preserve `source_view_ids`, `source_anchor_ids`, and `source_approach_ids` for every representative.
5. Run focused and full tests.
6. Commit as `feat: merge redundant grasp poses in SE3`.

## Task 3: Centralize configuration, determinism, and final schema

**Files:**
- Create: `grasp_config.py`
- Create: `grasp_determinism.py`
- Create: `grasp_schema.py`
- Modify: `multi_view_grasp.py`
- Test: `tests/test_grasp_config.py`
- Test: `tests/test_grasp_determinism.py`
- Test: `tests/test_grasp_schema.py`

1. Write failing tests for frozen defaults, seed propagation to Python/NumPy/Open3D, required final fields, quaternion ordering, finite-value rejection, and provenance fields.
2. Implement immutable `GraspGenerationConfig` with frozen defaults: cone mode, 15-degree cone, four azimuth samples plus nominal, KNN 30, depth samples 16, 5 mm/10 degree merge, deterministic seed 0. Keep `num_views` and `anchors_per_view` configurable.
3. Implement deterministic setup and invoke it once before a complete pipeline run.
4. Implement serialization-safe normalization containing translation, rotation matrix, quaternion xyzw, opening/depth in mm, total/subscores, geometric metadata, and provenance sets.
5. Pass centralized top-level parameters into multi-view generation while keeping current legacy defaults compatible.
6. Run focused and full tests.
7. Commit as `feat: standardize deterministic grasp annotations`.

## Task 4: Add final pipeline and three-file export

**Files:**
- Create: `grasp_pipeline.py`
- Create: `grasp_export.py`
- Create: `main.py`
- Test: `tests/test_grasp_pipeline.py`
- Test: `tests/test_grasp_export.py`
- Test: `tests/test_main_cli.py`

1. Write failing orchestration tests that assert the exact stage order and separate `raw_grasps` from `unique_grasps`.
2. Write failing export tests for `grasps.json`, `grasps.npz`, and `meta.json`, including shapes, finite values, counts, config, input scale, and timings.
3. Implement `run_grasp_annotation()` to prepare once, configure determinism, generate raw candidates, score/rank, symmetry-merge, normalize, and produce metadata.
4. Implement export using only the three requested files. JSON contains final unique grasps; NPZ contains dense numeric arrays without pickle objects; meta contains parameters, counts, and stage/total timings.
5. Implement CLI compatible with `python main.py --object ... --views 5 --anchors 3 --mode cone --output ...`.
6. Run focused and full tests.
7. Commit as `feat: export finalized grasp annotations`.

## Task 5: Reproducibility and freeze acceptance

**Files:**
- Create: `tests/test_finalization_acceptance.py`
- Create: `scripts/validate_grasp_freeze.py`
- Modify: `README.md` only if it already documents execution; otherwise create a concise usage section in the validation script help and avoid unrelated documentation changes.

1. Add fast automated acceptance tests for no NaNs, raw/unique count invariants, score ordering, deterministic equality, export integrity, and full-cloud collision path retention using controlled fixtures/mocks.
2. Add a real-model validation runner with cases `juxing`, `yuanzhu`, `shuilongtou` at `3 views x 2 anchors x 5 approaches`, plus one `5 x 3 x 5` case. It must run each configured case repeatedly and compare grasp counts, Top-K poses, and scores.
3. Run the complete unit suite under the requested py310 environment.
4. Run real-model smoke/acceptance cases available in `model/`, capturing logs and reporting any model that cannot meet the acceptance gate. Do not tag on partial or failed evidence.
5. Review the complete diff and confirm the user's unrelated local files remain unstaged/unmodified.
6. If every required real-model check passes, create annotated tag `v1.0-grasp-annotation`; otherwise leave the repository untagged and report the exact blocking evidence.
7. Push commits and tag (only if created) to the configured GitHub remote.

