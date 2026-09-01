# Ours / GN-Full common evaluation

This protocol keeps the frozen Ours generator and V4 ranking unchanged.  It
only converts the exported Ours poses into the official GraspNet/Dex-Net
convention and evaluates them against the same reference OBJ/SDF used by
GN-Full.

## Fixed GN subset

Create a deterministic 10,000-candidate subset from geometry-valid IDs:

```powershell
$env:PYTHONPATH=(Get-Location).Path
F:\Miniconda\envs\graspnet_baseline_py39_clean\Scripts\python.exe `
  scripts/common_eval/sample_gn_subset.py `
  --geometry-run results\graspnet-baseline\model2-full-geometry `
  --output results\graspnet-baseline\model2-full-fc-subset10k `
  --target-size 10000 --seed 0
```

Run the subset force-closure shards in fresh processes (the full GN-Full
executor can continue independently):

```powershell
F:\Miniconda\envs\graspnet_baseline_py39_clean\Scripts\python.exe `
  scripts/common_eval/run_force_closure_subset.py `
  --geometry-run results\graspnet-baseline\model2-full-geometry `
  --sdf-prefix C:\path\to\2_repaired `
  --candidate-ids results\graspnet-baseline\model2-full-fc-subset10k\sampled_candidate_ids.npy `
  --shard-dir results\graspnet-baseline\model2-full-fc-subset10k\shards `
  --shard-size 100 --workers 4

F:\Miniconda\envs\graspnet_baseline_py39_clean\Scripts\python.exe `
  scripts/common_eval/merge_force_closure_subset.py `
  --geometry-run results\graspnet-baseline\model2-full-geometry `
  --shard-dir results\graspnet-baseline\model2-full-fc-subset10k\shards `
  --candidate-ids results\graspnet-baseline\model2-full-fc-subset10k\sampled_candidate_ids.npy `
  --output results\graspnet-baseline\model2-full-fc-subset10k\merged
```

The merge is strict: duplicate, missing, extra, error, unscored, or
non-finite rows abort the run.

### Weighted stratified aggregation

The 10k subset is balanced by grasp point rather than by the number of
geometry-valid candidates at each point.  Aggregate it against the complete
geometry pass before using it as a full-run estimate:

```powershell
$env:PYTHONPATH=(Get-Location).Path
F:\Miniconda\envs\py310\python.exe `
  scripts/common_eval/summarize_stratified_subset.py `
  --geometry-run results\graspnet-baseline\model2-full-geometry `
  --subset-run results\graspnet-baseline\model2-full-fc-subset10k\merged `
  --output-dir results\common-eval\model2-gn-full-stratified
```

The JSON/CSV retain unweighted subset audit metrics and add candidate-weighted
FC/HQ rates, weighted mean :math:`\mu`, and full-population estimates.  The
CLI validates that point population sizes sum to the exact full geometry-valid
count and that sampled sizes sum to the manifest size.

## Ours adapter

The adapter evaluates every Ours record in input order.  It preserves the
native V4 score and never selects or re-ranks by the common `mu` score:

```powershell
F:\Miniconda\envs\py39\python.exe `
  scripts/common_eval/ours_official_common_eval.py `
  --ours-results results\ours-main\model2 `
  --ours-object model\2.ply `
  --reference-obj baselines\graspnet_annotation\assets\2\2_repaired.obj `
  --sdf-prefix C:\path\to\2_repaired `
  --output results\common-eval\model2-ours-corrected
```

Ours records are in millimetres and the Ours object frame.  The adapter
applies the recorded `T_object_world`, maps the local axes to the official
Dex-Net convention, and uses the final closure-refined width.

## Comparison table

```powershell
F:\Miniconda\envs\py39\python.exe scripts/common_eval/build_comparison.py `
  --gn-np20-geometry results\graspnet-baseline\model2-np20-geometry `
  --gn-np20-complete results\graspnet-baseline\model2-np20-complete-v2 `
  --gn-np20-shards results\graspnet-baseline\model2-np20-fc-shards `
  --gn-full-geometry results\graspnet-baseline\model2-full-geometry `
  --gn-full-subset results\graspnet-baseline\model2-full-fc-subset10k\merged `
  --gn-full-subset-shards results\graspnet-baseline\model2-full-fc-subset10k\shards `
  --gn-full-stats results\common-eval\model2-gn-full-stratified\stratified_statistics.json `
  --ours-common-summary results\common-eval\model2-ours-corrected\summary.json `
  --output-csv results\common-eval\model2-comparison.csv
```

Rows derived from the 10k subset include explicit `estimated_*` columns.  They
are extrapolations using the exact full geometry-valid ratio and must not be
reported as exact full force-closure counts.

## Multi-object final aggregation

Before running any additional object, inventory its paired reference assets:

```powershell
$env:PYTHONPATH=(Get-Location).Path
F:\Miniconda\envs\graspnet_baseline_py39_clean\Scripts\python.exe `
  scripts/common_eval/prepare_evaluation_assets.py `
  --manifest configs/formal_evaluation_objects.json `
  --output-dir results\common-eval\asset-inventory
```

The inventory classifies each object as `A_ready`, `A_mesh_needs_sdf`,
`B_repair_and_sdf`, or `C_reference_required`.  It never reconstructs a mesh,
uses a convex hull, or reuses another object's SDF.  Only an `A_ready` asset
may enter the frozen comparison protocol.

For the formal same-surface-input protocol, run the fixed reconstruction
pipeline on the selected PLY objects:

```powershell
F:\Miniconda\envs\py310\python.exe `
  scripts/common_eval/reconstruct_ply_reference.py `
  --manifest configs\formal_evaluation_objects.json `
  --output-root baselines\graspnet_annotation\assets\reconstructed `
  --object model2 --object juxing --object shuilongtou `
  --object yuanzhu --object huixing --object cat
```

Each object receives `reference_reconstructed.obj` and `asset_report.json`.
The report must show `watertight=true`, acceptable normalized p95 surface
error, and an existing 100³/5-padding SDF before the object is admitted to
GN-style evaluation.  If the fixed reconstruction fails these gates, report
the object as unavailable rather than tuning reconstruction parameters for it.

After each object has its own corrected comparison CSV, aggregate without
rerunning either method:

```powershell
$env:PYTHONPATH=(Get-Location).Path
F:\Miniconda\envs\py310\python.exe `
  scripts/common_eval/aggregate_object_comparisons.py `
  --object-csv model2=results\common-eval\model2-comparison-corrected.csv `
  --output-dir results\common-eval\all-objects
```

Repeat `--object-csv OBJECT=CSV` for every completed object.  The detail file
is `comparison_all_objects.csv`; `comparison_summary.csv` reports per-method
mean and population standard deviation for raw candidates, FC/HQ yield, HQ
rate, mean :math:`\mu`, and native runtime.  GN rows backed by the fixed
10k protocol are labelled `weighted_stratified_10k`; only a completed exact
run is labelled `full_exact`.
