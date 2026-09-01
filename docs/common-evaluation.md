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
  --output results\common-eval\model2-ours
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
  --ours-common-summary results\common-eval\model2-ours\summary.json `
  --output-csv results\common-eval\model2-comparison.csv
```

Rows derived from the 10k subset include explicit `estimated_*` columns.  They
are extrapolations using the exact full geometry-valid ratio and must not be
reported as exact full force-closure counts.
