param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repo = (
    Resolve-Path (
        Join-Path $PSScriptRoot ".."
    )
).Path

Set-Location $repo

$python = (
    Get-Command python -ErrorAction Stop
).Source

Write-Host "Repository: $repo"
Write-Host "Python:     $python"


# ============================================================
# Locate NVIDIA pip DLL directories inside active environment
# ============================================================

$venvRoot = Split-Path -Parent (
    Split-Path -Parent $python
)

$nvidiaRoot = Join-Path `
    $venvRoot `
    "Lib\site-packages\nvidia"

if (-not (Test-Path $nvidiaRoot)) {
    throw "NVIDIA pip directory not found: $nvidiaRoot"
}

$dllDirs = @(
    Get-ChildItem `
        -Path $nvidiaRoot `
        -Recurse `
        -File `
        -Filter *.dll |
    ForEach-Object {
        $_.DirectoryName
    } |
    Sort-Object -Unique
)

if ($dllDirs.Count -eq 0) {
    throw "No NVIDIA pip DLL directories found."
}

$env:PATH = (($dllDirs -join ";") + ";" + $env:PATH)

Write-Host (
    "NVIDIA DLL directories added to PATH: " +
    $dllDirs.Count
)


# ============================================================
# CUDA/OpenMM smoke test
# ============================================================

$smokeScript = Join-Path `
    $repo `
    "scripts\cuda_smoke_test.py"

& $python $smokeScript

if ($LASTEXITCODE -ne 0) {
    throw "CUDA/OpenMM smoke test failed."
}


# ============================================================
# Queue
# ============================================================

$queue = Join-Path `
    $repo `
    "scripts\run_drop_ensemble_queue.py"

$queueArgs = @(
    $queue
)

if ($DryRun) {
    $queueArgs += "--dry-run"
}


if ($DryRun) {

    Write-Host ""
    Write-Host "Running queue dry-run..."

    & $python @queueArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Queue dry-run failed."
    }

    exit 0
}


$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

$logDir = Join-Path `
    $repo `
    "analysis\ensemble"

New-Item `
    -ItemType Directory `
    -Force `
    -Path $logDir |
Out-Null

$log = Join-Path `
    $logDir `
    "drop_ensemble_queue_$stamp.log"

Write-Host ""
Write-Host "Starting Drop02-Drop13 queue."
Write-Host "Log: $log"
Write-Host ""

& $python @queueArgs 2>&1 |
    Tee-Object -FilePath $log

$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    throw (
        "Queue stopped with exit code $exitCode. " +
        "Inspect: $log"
    )
}

Write-Host ""
Write-Host "DROP ENSEMBLE QUEUE FINISHED SUCCESSFULLY"
