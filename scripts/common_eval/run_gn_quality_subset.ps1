param(
    [Parameter(Mandatory = $true)][string]$GeometryRun,
    [Parameter(Mandatory = $true)][string]$SdfPrefix,
    [Parameter(Mandatory = $true)][string]$Output,
    [int]$TargetSize = 10000,
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"
$repo = (Get-Location).Path
$env:PYTHONPATH = $repo
$py39 = "F:\Miniconda\envs\graspnet_baseline_py39_clean\Scripts\python.exe"
$py310 = "F:\Miniconda\envs\py310\python.exe"
$out = [IO.Path]::GetFullPath($Output)
$subset = Join-Path $out "subset10k"
$shards = Join-Path $subset "shards"
$merged = Join-Path $subset "merged"
$stats = Join-Path $out "stratified"
New-Item -ItemType Directory -Force -Path $out | Out-Null

& $py310 scripts/common_eval/sample_gn_subset.py --geometry-run $GeometryRun --output $subset --target-size $TargetSize --seed 0
& $py39 scripts/common_eval/run_force_closure_subset.py --geometry-run $GeometryRun --sdf-prefix $SdfPrefix --candidate-ids (Join-Path $subset "sampled_candidate_ids.npy") --shard-dir $shards --shard-size 100 --workers $Workers
& $py39 scripts/common_eval/merge_force_closure_subset.py --geometry-run $GeometryRun --shard-dir $shards --candidate-ids (Join-Path $subset "sampled_candidate_ids.npy") --output $merged
& $py310 scripts/common_eval/summarize_stratified_subset.py --geometry-run $GeometryRun --subset-run $merged --output-dir $stats --hq-threshold 0.4
