$ErrorActionPreference = "Stop"
$repo = (Get-Location).Path
$env:PYTHONPATH = $repo
$python = "F:\Miniconda\envs\graspnet_baseline_py39_clean\Scripts\python.exe"

# model2 is launched separately because it is the largest PLY geometry run.
while (-not (Test-Path "results/graspnet-baseline/model2-ply-geometry/summary.json")) {
    Start-Sleep -Seconds 30
}

$jobs = @(
    @{ name = "shuilongtou"; ply = "model/shuilongtou.ply"; unit = "mm" },
    @{ name = "yuanzhu"; ply = "model/yuanzhu.ply"; unit = "mm" },
    @{ name = "huixing"; ply = "model/huixing.ply"; unit = "mm" },
    @{ name = "cat"; ply = "model/colmap/cat.ply"; unit = "m" }
)
foreach ($job in $jobs) {
    $output = "results/graspnet-baseline/$($job.name)-ply-geometry"
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    $summary = Join-Path $output "summary.json"
    if (Test-Path $summary) {
        Write-Host "Skipping completed $($job.name)"
        continue
    }
    $log_dir = "results/graspnet-baseline/logs"
    New-Item -ItemType Directory -Force -Path $log_dir | Out-Null
    $log = Join-Path $log_dir "$($job.name)-geometry-only.log"
    Write-Host "Running $($job.name) from $($job.ply)"
    & $python -u -m baselines.graspnet_annotation.run_graspnet_baseline `
        --surface-ply $job.ply --input-unit $job.unit --output $output --geometry-only *> $log
    if ($LASTEXITCODE -ne 0) {
        throw "Geometry-only failed for $($job.name); see $log"
    }
}
