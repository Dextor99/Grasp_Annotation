# Grasp annotation v1.1

`v1.1-grasp-annotation` is the finalized version of the current multi-view
surface-conditioned grasp generator. The generation, scoring, collision, and
SE(3) merge rules are frozen.

The exported widths have two meanings:

- `search_opening_mm` (`opening_mm` for legacy consumers): collision-safe
  aperture used during candidate search.
- `grasp_width_mm`: final local support span plus closure margin, capped by the
  search aperture.

After local closure refinement, the original search aperture is rebuilt at the
refined pose and checked against the full object cloud. Only candidates that
pass this post-refinement safety check are exported.

## Experiment statistics

Run the generator with `main.py`, then summarize one or more result directories:

```powershell
F:\Miniconda\envs\py310\python.exe scripts\summarize_grasp_results.py `
  results\freeze-validation\cat_v3_a2 `
  results\freeze-validation\juxing_v3_a2 `
  --output results\freeze-validation\experiment_summary.csv
```

The summary distinguishes generated candidates, scored candidates, refinement
inputs, geometry rejections, post-refinement collision rejections, closure
valid grasps, and unique merged grasps. `raw_grasp_count` remains as a legacy
alias for the closure-valid pre-merge count.
