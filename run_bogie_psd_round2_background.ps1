param(
    [switch]$CompareOnly
)

$ErrorActionPreference = "Stop"
$Workspace = $PSScriptRoot
Set-Location $Workspace

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $Workspace "results\bogie_psd_round2_scan\_comparison"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$StdoutLog = Join-Path $LogDir "round2_background_$Timestamp.out.log"
$StderrLog = Join-Path $LogDir "round2_background_$Timestamp.err.log"
$PidFile = Join-Path $LogDir "round2_background_$Timestamp.pid.txt"

$ArgsList = @(
    "run_sweep_and_compare.py",
    "--manifest",
    "configs\sweeps\high_speed_passenger_bogie_psd_round2_scan.yaml",
    "--skip-single-analysis"
)

if ($CompareOnly) {
    $ArgsList += "--compare-only"
}

$Process = Start-Process `
    -FilePath "python" `
    -ArgumentList $ArgsList `
    -WorkingDirectory $Workspace `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Hidden `
    -PassThru

@(
    "pid=$($Process.Id)",
    "started_at=$(Get-Date -Format s)",
    "stdout=$StdoutLog",
    "stderr=$StderrLog",
    "compare_only=$CompareOnly"
) | Set-Content -Path $PidFile -Encoding UTF8

Write-Host "Started round2 sweep in background."
Write-Host "PID: $($Process.Id)"
Write-Host "stdout: $StdoutLog"
Write-Host "stderr: $StderrLog"
Write-Host "pid file: $PidFile"
