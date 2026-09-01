# GraspNet-1Billion-Style Dense Annotation Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, deterministic GN-Full dense annotation baseline that emits GraspNet-compatible raw object labels and force-closure scores without modifying the frozen ours pipeline.

**Architecture:** The baseline is a new `baselines.graspnet_annotation` package.  Pure NumPy modules own configuration, unit conversion, Fibonacci topology, streaming index expansion, and serialization; an explicit adapter is the only module allowed to import `graspnetAPI` or Dex-Net.  This makes 18/14,400/288,000-candidate topology tests runnable without native geometry dependencies, while full force-closure runs fail closed if official prerequisites are unavailable.

**Tech Stack:** Python 3.10, NumPy, trimesh, pytest/unittest compatibility, official `graspnetAPI`, Dex-Net-compatible model/SDF tooling, Open3D only when required by the official API.

---

## File structure

| Path | Responsibility |
| --- | --- |
| `baselines/__init__.py` | Marks independent baseline namespace. |
| `baselines/graspnet_annotation/__init__.py` | Exposes the public config and runner entry points only. |
| `baselines/graspnet_annotation/config.py` | Immutable GN-Full parameters, provenance, and validation. |
| `baselines/graspnet_annotation/preprocess.py` | Load mesh, apply explicit metre conversion, and report mesh/SDF readiness. |
| `baselines/graspnet_annotation/view_sampling.py` | Official-order Fibonacci viewpoints and in-plane angle/depth offsets. |
| `baselines/graspnet_annotation/candidate_generation.py` | Bounded-memory point-batch index expansion. |
| `baselines/graspnet_annotation/label_arrays.py` | Raw GraspNet-format arrays and compact valid-grasp conversion. |
| `baselines/graspnet_annotation/official_adapter.py` | Runtime dependency gate and official collision/Dex-Net force-closure calls. |
| `baselines/graspnet_annotation/export.py` | Safe output directory validation and deterministic NPZ/JSON/CSV output. |
| `baselines/graspnet_annotation/run_graspnet_baseline.py` | CLI orchestration. |
| `baselines/graspnet_annotation/requirements.txt` | Baseline-only pinned/declared dependency set and source-install instructions. |
| `tests/baselines/test_*.py` | Isolated unit, CLI, and optional real-backend integration tests. |
| `docs/graspnet-baseline.md` | Invocation, physical-unit contract, prerequisites, and output semantics. |

### Task 1: Create the isolated configuration contract

**Files:**
- Create: `baselines/__init__.py`
- Create: `baselines/graspnet_annotation/__init__.py`
- Create: `baselines/graspnet_annotation/config.py`
- Test: `tests/baselines/test_graspnet_config.py`

- [ ] **Step 1: Write the failing configuration tests**

```python
from baselines.graspnet_annotation.config import DenseAnnotationConfig


def test_full_defaults_match_graspnet_label_topology():
    config = DenseAnnotationConfig.full()
    assert (config.num_views, config.num_angles, config.depths_m) == (300, 12, (0.01, 0.02, 0.03, 0.04))
    assert config.candidates_per_point == 14_400
    assert config.input_unit == "m"


def test_debug_override_has_exact_cartesian_candidate_count():
    config = DenseAnnotationConfig.full(num_views=3, num_angles=3, depths_m=(0.01, 0.02))
    assert config.candidates_per_point == 18


def test_rejects_unknown_unit_and_nonpositive_gripper_values():
    with pytest.raises(ValueError, match="input_unit"):
        DenseAnnotationConfig.full(input_unit="unit_sphere")
    with pytest.raises(ValueError, match="max_width_m"):
        DenseAnnotationConfig.full(max_width_m=0.0)
```

- [ ] **Step 2: Run the tests and confirm they fail because the package is absent**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_config.py -q`

Expected: `ModuleNotFoundError: No module named 'baselines'`.

- [ ] **Step 3: Implement the immutable configuration**

```python
@dataclass(frozen=True)
class DenseAnnotationConfig:
    input_unit: str = "m"
    seed: int = 0
    surface_samples: int = 6000
    voxel_size_m: float = 0.006
    max_grasp_points: int = 1200
    num_views: int = 300
    num_angles: int = 12
    depths_m: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04)
    height_m: float = 0.02
    depth_base_m: float = 0.02
    finger_width_m: float = 0.01
    max_width_m: float = 0.12
    empty_thresh: int = 10
    collision_margin_m: float = 0.004
    friction_coefficients: tuple[float, ...] = tuple(np.arange(1.0, 0.0, -0.1).round(1))

    @property
    def candidates_per_point(self) -> int:
        return self.num_views * self.num_angles * len(self.depths_m)
```

Validate all physical values as positive, enforce metres as the internal unit, and expose `parameter_provenance()` with `official_topology`, `public_reference_default`, and `local_baseline_config` groups.

- [ ] **Step 4: Run the configuration tests**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_config.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the configuration contract**

```powershell
git add baselines/__init__.py baselines/graspnet_annotation/__init__.py baselines/graspnet_annotation/config.py tests/baselines/test_graspnet_config.py
git commit -m "feat: add graspnet dense baseline config"
```

### Task 2: Add geometry readiness and explicit unit conversion

**Files:**
- Create: `baselines/graspnet_annotation/preprocess.py`
- Test: `tests/baselines/test_graspnet_preprocess.py`

- [ ] **Step 1: Write the failing preprocessing tests**

```python
from baselines.graspnet_annotation.preprocess import convert_vertices_to_meters, validate_mesh_readiness


def test_converts_millimetre_vertices_to_metres_once():
    vertices = np.array([[0.0, 0.0, 0.0], [100.0, 20.0, 10.0]])
    converted = convert_vertices_to_meters(vertices, "mm")
    np.testing.assert_allclose(converted[1], [0.1, 0.02, 0.01])


def test_rejects_missing_sdf_for_force_closure_ready_run(tmp_path):
    mesh = tmp_path / "object.stl"
    mesh.write_bytes(b"solid empty\nendsolid empty\n")
    with pytest.raises(FileNotFoundError, match="SDF"):
        validate_mesh_readiness(mesh, sdf_path=None, require_sdf=True)
```

- [ ] **Step 2: Run the tests and confirm import failure**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_preprocess.py -q`

Expected: import failure for `preprocess`.

- [ ] **Step 3: Implement conversion and readiness result**

```python
UNIT_TO_METRES = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}

def convert_vertices_to_meters(vertices: np.ndarray, input_unit: str) -> np.ndarray:
    if input_unit not in UNIT_TO_METRES:
        raise ValueError(f"input_unit must be one of {sorted(UNIT_TO_METRES)}, got {input_unit!r}")
    return np.asarray(vertices, dtype=np.float64) * UNIT_TO_METRES[input_unit]

def validate_mesh_readiness(mesh_path: Path, sdf_path: Path | None, require_sdf: bool) -> MeshReadiness:
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    if require_sdf and (sdf_path is None or not sdf_path.is_file()):
        raise FileNotFoundError("SDF is required for Dex-Net force-closure; provide --sdf or generate one first")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Mesh has no triangle geometry: {mesh_path}")
    return MeshReadiness(is_watertight=bool(mesh.is_watertight), vertex_count=len(mesh.vertices), face_count=len(mesh.faces))
```

`load_mesh_in_metres()` must return a copied mesh with converted vertices and metadata recording original unit and scale.  It must never write over the source mesh.

- [ ] **Step 4: Run preprocessing tests**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_preprocess.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit preprocessing**

```powershell
git add baselines/graspnet_annotation/preprocess.py tests/baselines/test_graspnet_preprocess.py
git commit -m "feat: validate graspnet baseline geometry inputs"
```

### Task 3: Reproduce the 300 × 12 × 4 topology and stream candidate indices

**Files:**
- Create: `baselines/graspnet_annotation/view_sampling.py`
- Create: `baselines/graspnet_annotation/candidate_generation.py`
- Test: `tests/baselines/test_graspnet_candidate_generation.py`

- [ ] **Step 1: Write the failing topology tests**

```python
from baselines.graspnet_annotation.candidate_generation import iter_candidate_batches
from baselines.graspnet_annotation.config import DenseAnnotationConfig
from baselines.graspnet_annotation.view_sampling import generate_views, make_offsets


def test_one_point_official_topology_has_14400_indexed_candidates():
    config = DenseAnnotationConfig.full()
    batches = list(iter_candidate_batches(np.zeros((1, 3)), config, point_batch_size=1))
    assert sum(batch.size for batch in batches) == 14_400
    assert batches[0].view_ids.min() == 0 and batches[0].view_ids.max() == 299
    assert batches[0].angle_ids.min() == 0 and batches[0].angle_ids.max() == 11
    assert batches[0].depth_ids.min() == 0 and batches[0].depth_ids.max() == 3


def test_debug_topology_is_exactly_18_candidates():
    config = DenseAnnotationConfig.full(num_views=3, num_angles=3, depths_m=(0.01, 0.02))
    assert next(iter_candidate_batches(np.zeros((1, 3)), config)).size == 18


def test_twenty_point_streaming_never_expands_more_than_one_point_batch():
    config = DenseAnnotationConfig.full()
    batches = list(iter_candidate_batches(np.zeros((20, 3)), config, point_batch_size=1))
    assert len(batches) == 20
    assert sum(batch.size for batch in batches) == 288_000
    assert max(batch.size for batch in batches) == 14_400
```

- [ ] **Step 2: Run and confirm failure**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_candidate_generation.py -q`

Expected: import failure for candidate modules.

- [ ] **Step 3: Implement deterministic view and batch expansion**

```python
def generate_views(num_views: int) -> np.ndarray:
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    indices = np.arange(num_views, dtype=np.float64)
    z = 1.0 - (2.0 * indices + 1.0) / num_views
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = 2.0 * np.pi * indices * phi
    return np.stack((radius * np.cos(theta), radius * np.sin(theta), z), axis=1).astype(np.float32)

def make_offsets(config: DenseAnnotationConfig) -> np.ndarray:
    angles = np.arange(config.num_angles, dtype=np.float32) * (np.pi / config.num_angles)
    offsets = np.zeros((config.num_views, config.num_angles, len(config.depths_m), 3), dtype=np.float32)
    offsets[..., 0] = angles[None, :, None]
    offsets[..., 1] = np.asarray(config.depths_m, dtype=np.float32)[None, None, :]
    return offsets
```

`CandidateBatch` must carry `point_indices`, `view_ids`, `angle_ids`, `depth_ids`, `points_m`, `views`, and the raw `offsets`.  It must use `np.meshgrid(..., indexing="ij")` and flatten in `(point, view, angle, depth)` order so it writes directly into the official raw-label axes.

- [ ] **Step 4: Run topology tests**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_candidate_generation.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit topology generation**

```powershell
git add baselines/graspnet_annotation/view_sampling.py baselines/graspnet_annotation/candidate_generation.py tests/baselines/test_graspnet_candidate_generation.py
git commit -m "feat: generate streamed graspnet candidate topology"
```

### Task 4: Add raw-label storage and safe export

**Files:**
- Create: `baselines/graspnet_annotation/label_arrays.py`
- Create: `baselines/graspnet_annotation/export.py`
- Test: `tests/baselines/test_graspnet_export.py`

- [ ] **Step 1: Write failing raw-label and export tests**

```python
def test_raw_label_arrays_have_official_axes_and_invalid_score_sentinel():
    labels = RawLabelArrays.create(point_count=1, config=DenseAnnotationConfig.full())
    assert labels.points.shape == (1, 3)
    assert labels.offsets.shape == (1, 300, 12, 4, 3)
    assert labels.collision.shape == (1, 300, 12, 4)
    assert np.all(labels.scores == -1.0)


def test_export_refuses_to_mix_with_existing_user_files(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "keep.txt").write_text("user file", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_annotation_run(output, labels=labels, summary={"units": "m"}, timing_rows=[])
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user file"
```

- [ ] **Step 2: Run and confirm failure**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_export.py -q`

Expected: import failure for `label_arrays` and `export`.

- [ ] **Step 3: Implement raw arrays, compact valid grasps, and atomic export**

```python
@dataclass
class RawLabelArrays:
    points: np.ndarray
    offsets: np.ndarray
    collision: np.ndarray
    scores: np.ndarray

    @classmethod
    def create(cls, point_count: int, config: DenseAnnotationConfig) -> "RawLabelArrays":
        shape = (point_count, config.num_views, config.num_angles, len(config.depths_m))
        return cls(
            points=np.zeros((point_count, 3), dtype=np.float32),
            offsets=np.zeros((*shape, 3), dtype=np.float32),
            collision=np.ones(shape, dtype=bool),
            scores=np.full(shape, -1.0, dtype=np.float32),
        )
```

`write_point_result()` writes a completed point's widths, collision values, and minimum-friction scores into fixed axes. `to_valid_grasps()` returns `float32` `(K, 17)` records in documented order `[quality, width, height, depth, rotation_3x3_row_major, translation_3, object_id]`, with `quality = 1.1 - mu_min` only for valid `0 < mu_min <= 1.0` entries. `export_annotation_run()` writes exactly `grasp_labels.npz`, `valid_grasps.npy`, `summary.json`, `timing.csv`, and `run_config.json` into a new or empty directory after all validation passes; it must not delete or overwrite arbitrary existing files.

- [ ] **Step 4: Run export tests**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_export.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit label storage/export**

```powershell
git add baselines/graspnet_annotation/label_arrays.py baselines/graspnet_annotation/export.py tests/baselines/test_graspnet_export.py
git commit -m "feat: export graspnet style raw annotation labels"
```

### Task 5: Install and isolate official runtime prerequisites

**Files:**
- Create: `baselines/graspnet_annotation/requirements.txt`
- Create: `baselines/graspnet_annotation/official_adapter.py`
- Test: `tests/baselines/test_graspnet_official_adapter.py`
- Modify: `docs/graspnet-baseline.md`

- [ ] **Step 1: Write failing dependency-gate tests**

```python
from baselines.graspnet_annotation.official_adapter import OfficialBackendUnavailable, require_official_backend


def test_missing_backend_gives_actionable_install_message(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    with pytest.raises(OfficialBackendUnavailable, match="graspnetAPI.*Dex-Net"):
        require_official_backend()


def test_backend_gate_returns_imported_symbols_when_available(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())
    sentinel = object()
    monkeypatch.setattr(importlib, "import_module", lambda _: sentinel)
    assert require_official_backend() == (sentinel, sentinel)
```

- [ ] **Step 2: Run and confirm failure**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_official_adapter.py -q`

Expected: import failure for `official_adapter`.

- [ ] **Step 3: Implement a fail-closed official backend gate**

```python
class OfficialBackendUnavailable(RuntimeError):
    pass

def require_official_backend() -> tuple[ModuleType, ModuleType]:
    required = {"graspnetAPI": "graspnetAPI", "dexnet": "dexnet.grasping"}
    missing = [package for package in required if importlib.util.find_spec(package) is None]
    if missing:
        raise OfficialBackendUnavailable(
            "GN-Full requires graspnetAPI and Dex-Net for official-compatible collision/force closure. "
            "Install baselines/graspnet_annotation/requirements.txt, then retry. Missing: " + ", ".join(missing)
        )
    return tuple(importlib.import_module(name) for name in required.values())
```

Keep imports lazy so every topology/export test can run without native packages.  Create a requirements file that pins only packages proven importable under the project `py310` interpreter; record source-install commands for the official API and any required SDF generator rather than silently vendoring third-party code into this repository.

- [ ] **Step 4: Install the approved runtime in the `py310` environment and record versions**

Run first: `F:\Miniconda\envs\py310\python.exe -m pip install -r baselines\graspnet_annotation\requirements.txt`

Then verify: `F:\Miniconda\envs\py310\python.exe -c "import graspnetAPI; import dexnet.grasping; print('official backend ready')"`

Expected: `official backend ready`.  If a package is incompatible with Python 3.10, preserve the installer output, mark the integration test skipped with the exact reason, and stop before claiming GN-Full force-closure reproduction.

- [ ] **Step 5: Run backend-gate tests**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_official_adapter.py -q`

Expected: `2 passed`.

- [ ] **Step 6: Commit dependency isolation**

```powershell
git add baselines/graspnet_annotation/requirements.txt baselines/graspnet_annotation/official_adapter.py tests/baselines/test_graspnet_official_adapter.py docs/graspnet-baseline.md
git commit -m "feat: add graspnet official backend adapter"
```

### Task 6: Implement mesh sampling, official collision, and force-closure evaluation

**Files:**
- Create: `baselines/graspnet_annotation/grasp_point_sampling.py`
- Modify: `baselines/graspnet_annotation/official_adapter.py`
- Test: `tests/baselines/test_graspnet_sampling.py`
- Test: `tests/baselines/test_graspnet_official_integration.py`

- [ ] **Step 1: Write failing sampling and adapter-contract tests**

```python
def test_seeded_surface_then_voxel_sampling_is_repeatable_and_capped():
    mesh = trimesh.creation.box(extents=(0.1, 0.06, 0.04))
    config = DenseAnnotationConfig.full(surface_samples=200, max_grasp_points=20)
    first = sample_grasp_points(mesh, config)
    second = sample_grasp_points(mesh, config)
    np.testing.assert_allclose(first, second)
    assert first.shape == (20, 3)


def test_evaluation_result_rejects_nonfinite_mu_and_preserves_collision_shape():
    with pytest.raises(ValueError, match="finite"):
        PointEvaluation(collision=np.zeros((300, 12, 4), bool), mu_min=np.full((300, 12, 4), np.nan))
```

- [ ] **Step 2: Run and confirm failure**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_sampling.py -q`

Expected: import failure for `grasp_point_sampling` or `PointEvaluation`.

- [ ] **Step 3: Implement deterministic sampling and official evaluation adapter**

```python
def sample_grasp_points(mesh: trimesh.Trimesh, config: DenseAnnotationConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    faces = rng.choice(len(mesh.faces), size=config.surface_samples, replace=True)
    barycentric = rng.dirichlet((1.0, 1.0, 1.0), size=config.surface_samples)
    triangles = mesh.triangles[faces]
    samples = (triangles * barycentric[..., None]).sum(axis=1)
    return voxel_downsample_and_cap(samples, config.voxel_size_m, config.max_grasp_points)
```

`OfficialGraspNetBackend.evaluate_point()` must accept one sampled model-frame point and its `(300, 12, 4, 3)` offsets; obtain rotations using the official view-to-matrix utility, estimate width using the baseline's prescribed local geometry rule, invoke the official-compatible collision path, and compute `mu_min` using the configured descending friction sweep.  Its output is exactly `PointEvaluation(widths_m, collision, mu_min, rotations, translations)` with shape `(V, A, D)` for scalar fields.  Collision/empty candidates always receive raw score `-1.0`; valid candidates receive their finite `mu_min`.

- [ ] **Step 4: Add the optional real backend test**

```python
@pytest.mark.integration
def test_model_2_stl_debug_run_produces_finite_valid_scores():
    pytest.importorskip("graspnetAPI")
    pytest.importorskip("dexnet.grasping")
    result = run_debug_annotation(Path("model/2.stl"), input_unit="mm", num_views=3, num_angles=3, depths_m=(0.01, 0.02))
    assert result.raw_candidate_count == 18
    assert np.isfinite(result.labels.scores[result.labels.scores >= 0]).all()
```

This test must explicitly skip when dependencies or a generated SDF are absent; it must never replace a skip with fabricated force-closure values.

- [ ] **Step 5: Run unit tests, then integration readiness test**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_sampling.py -q`

Expected: `2 passed`.

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_official_integration.py -q -m integration`

Expected: either a finite 18-candidate result or a single explicit skip naming the missing official prerequisite.

- [ ] **Step 6: Commit sampling and official evaluation**

```powershell
git add baselines/graspnet_annotation/grasp_point_sampling.py baselines/graspnet_annotation/official_adapter.py tests/baselines/test_graspnet_sampling.py tests/baselines/test_graspnet_official_integration.py
git commit -m "feat: evaluate dense graspnet candidates"
```

### Task 7: Connect the GN-Full runner and CLI

**Files:**
- Create: `baselines/graspnet_annotation/run_graspnet_baseline.py`
- Test: `tests/baselines/test_graspnet_cli.py`
- Modify: `docs/graspnet-baseline.md`

- [ ] **Step 1: Write failing CLI orchestration tests**

```python
def test_cli_builds_full_config_and_delegates_to_runner(monkeypatch, tmp_path):
    from baselines.graspnet_annotation import run_graspnet_baseline as cli
    captured = {}
    monkeypatch.setattr(cli, "run_full_annotation", lambda **kwargs: captured.update(kwargs) or {"valid_count": 1})
    assert cli.main(["--mesh", "model/2.stl", "--input-unit", "mm", "--mode", "full", "--output", str(tmp_path / "out")]) == 0
    assert captured["config"].candidates_per_point == 14_400
    assert captured["mesh_path"] == Path("model/2.stl")


def test_cli_rejects_budget_mode_before_its_separate_plan_exists():
    from baselines.graspnet_annotation import run_graspnet_baseline as cli
    with pytest.raises(SystemExit):
        cli.main(["--mesh", "model/2.stl", "--mode", "budget", "--output", "results/x"])
```

- [ ] **Step 2: Run and confirm failure**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_cli.py -q`

Expected: import failure for the runner.

- [ ] **Step 3: Implement `run_full_annotation()` and CLI parsing**

```python
def run_full_annotation(*, mesh_path: Path, sdf_path: Path | None, config: DenseAnnotationConfig, output: Path) -> dict[str, int]:
    readiness = validate_mesh_readiness(mesh_path, sdf_path, require_sdf=True)
    backend = OfficialGraspNetBackend.from_paths(mesh_path, sdf_path, config)
    points = sample_grasp_points(load_mesh_in_metres(mesh_path, config.input_unit).mesh, config)
    labels = RawLabelArrays.create(len(points), config)
    for batch in iter_candidate_batches(points, config, point_batch_size=1):
        labels.write_point_result(batch.point_index, backend.evaluate_point(batch))
    return export_annotation_run(output, labels, build_summary(labels, readiness, config), timing_rows=backend.timings)
```

CLI accepts only `--mesh`, `--sdf`, `--input-unit {m,mm,cm}`, `--mode full`, `--output`, `--max-grasp-points`, and debug topology overrides.  It prints candidate count, valid count, valid rate, and output path.  It must check dependencies/SDF before creating the output directory.

- [ ] **Step 4: Run CLI tests**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_cli.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit the runner**

```powershell
git add baselines/graspnet_annotation/run_graspnet_baseline.py tests/baselines/test_graspnet_cli.py docs/graspnet-baseline.md
git commit -m "feat: add graspnet full baseline runner"
```

### Task 8: Run formal topology and geometry acceptance checks

**Files:**
- Modify: `docs/graspnet-baseline.md`
- Test: `tests/baselines/test_graspnet_acceptance.py`

- [ ] **Step 1: Write failing end-to-end acceptance assertions**

```python
def test_debug_topology_exports_18_candidates(tmp_path):
    report = run_with_fake_backend(tmp_path, num_points=1, num_views=3, num_angles=3, depths=(0.01, 0.02))
    assert report["raw_candidate_count"] == 18


def test_official_topology_exports_14400_candidates_for_one_point(tmp_path):
    report = run_with_fake_backend(tmp_path, num_points=1, num_views=300, num_angles=12, depths=(0.01, 0.02, 0.03, 0.04))
    assert report["raw_candidate_count"] == 14_400


def test_streaming_twenty_points_exports_288000_candidates_without_nan(tmp_path):
    report = run_with_fake_backend(tmp_path, num_points=20, num_views=300, num_angles=12, depths=(0.01, 0.02, 0.03, 0.04))
    assert report["raw_candidate_count"] == 288_000
    assert report["nonfinite_score_count"] == 0
```

- [ ] **Step 2: Run and confirm failure**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines/test_graspnet_acceptance.py -q`

Expected: failure because the fake-backend test seam is absent.

- [ ] **Step 3: Implement the deterministic fake-backend seam for topology-only tests**

Implement `run_full_annotation(..., backend_factory=OfficialGraspNetBackend.from_paths)` and a test-only `FakeBackend` that returns a fixed valid `mu_min=0.4`, non-collision mask, valid rotations, and translations.  Production CLI must not expose this seam.

- [ ] **Step 4: Run all baseline tests**

Run: `F:\Miniconda\envs\py310\python.exe -m pytest tests/baselines -q`

Expected: all pure-Python tests pass; official integration either passes or reports the exact prerequisite skip.

- [ ] **Step 5: Execute the first real mesh readiness command**

Run: `F:\Miniconda\envs\py310\python.exe baselines\graspnet_annotation\run_graspnet_baseline.py --mesh model\2.stl --input-unit mm --mode full --output results\baselines\graspnet\2_debug --num-views 3 --num-angles 3 --depths 0.01 0.02 --max-grasp-points 1`

Expected: either `raw_candidate_count=18` with finite valid scores and five export files, or a nonzero exit explaining the precise missing SDF/official dependency/watertightness requirement.  Do not call this run a completed GN-Full experiment unless it succeeds.

- [ ] **Step 6: Commit acceptance tests and documentation**

```powershell
git add tests/baselines/test_graspnet_acceptance.py docs/graspnet-baseline.md
git commit -m "test: verify graspnet dense baseline protocol"
```

## Plan self-review

- **Spec coverage:** isolation is enforced by package boundaries (Tasks 1–8); metres and mesh/SDF readiness are Task 2; official `300 × 12 × 4` topology and stream bounds are Task 3; raw standard fields and score semantics are Task 4; official dependency gate is Task 5; collision/force closure is Task 6; reproducible CLI is Task 7; 18, 14,400, and 288,000 acceptance checks are Task 8. GN-Fair/Budget is intentionally excluded and requires a later spec/plan after GN-Full validation.
- **Placeholder scan:** all implementation tasks name exact files, test names, commands, expected results, and public interfaces. There are no undeclared `TODO`/`TBD` tasks.
- **Type consistency:** `DenseAnnotationConfig`, `RawLabelArrays`, `CandidateBatch`, `PointEvaluation`, `OfficialGraspNetBackend`, `run_full_annotation`, and `export_annotation_run` use the same names and metric units throughout.
