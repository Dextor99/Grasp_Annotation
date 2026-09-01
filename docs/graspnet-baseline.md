# Independent GraspNet-style dense baseline

`baselines/graspnet_annotation` is an isolated object-level dense annotation
baseline. It does not modify or import the frozen `ours` grasp-generation
pipeline. Internal geometry units are metres and the official raw topology is
300 viewpoints × 12 in-plane angles × 4 depths = 14,400 candidates per point.

## Dedicated environment

Use the Python 3.9 environment because the official API bundles legacy
Dex-Net dependencies:

```powershell
F:\Miniconda\envs\graspnet_baseline_py39_clean\Scripts\python.exe -m pip install -r baselines\graspnet_annotation\requirements.txt
F:\Miniconda\envs\graspnet_baseline_py39_clean\Scripts\python.exe -m pip install --no-deps graspnetAPI==1.2.11
```

## Gate 2–4 validation

Before running a real object, validate the official topology, rotation
convention, collision semantics, and paired OBJ/SDF loader with the independent
watertight cube asset:

```powershell
$env:PYTHONPATH = (Get-Location).Path
F:\Miniconda\envs\graspnet_baseline_py39_clean\Scripts\python.exe `
  scripts\baselines\run_gates.py `
  --asset-dir baselines\graspnet_annotation\assets\debug_cube `
  --report-json results\graspnet-baseline\gates\debug_cube_gate_report.json
```

The expected report has `gate2_topology`, `gate3_pose_convention`,
`gate4_collision_geometry`, and `sdf_load` all set to `true`; the candidate
count is 14,400 and the SDF grid is 64³. Official collision's returned mask
includes empty grasps by design, so the report records the separate empty mask
as well.

The debug cube is independent test data. Original project meshes, including
`model/2.stl`, are never overwritten by the gate runner.

## SDF generation for an independent repaired asset

SDFGen is an external native utility; keep it outside this repository.  The
wrapper validates paths and invokes it without modifying the OBJ:

```powershell
F:\Miniconda\envs\graspnet_baseline_py39_clean\Scripts\python.exe `
  scripts\baselines\generate_sdf.py `
  --sdf-exe C:\path\to\sdf_gen.exe `
  --obj baselines\graspnet_annotation\assets\2\2_repaired.obj `
  --grid-dim 64 --padding 5
```
