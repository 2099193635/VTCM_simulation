param(
    [ValidateSet(150, 300)]
    [int]$TargetSeeds = 150,
    [ValidateRange(1, 32)]
    [int]$Workers = 3,
    [ValidateRange(1, 32)]
    [int]$ThreadsPerWorker = 1,
    [string]$PythonExe = "",
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

function Get-DatasetProcesses {
    try {
        return @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.Name -eq "python.exe" -and $_.CommandLine -like "*component_inverse_v2*"
        })
    }
    catch {
        Write-Warning "Unable to query Python command lines: $($_.Exception.Message)"
        return @()
    }
}

$targets = Get-DatasetProcesses
if ($targets.Count -gt 0) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class DatasetResume {
    [DllImport("ntdll.dll")]
    public static extern int NtResumeProcess(IntPtr handle);
}
"@ -ErrorAction SilentlyContinue

    foreach ($target in $targets) {
        try {
            $process = Get-Process -Id $target.ProcessId -ErrorAction Stop
            [void][DatasetResume]::NtResumeProcess($process.Handle)
            Write-Host "Resumed PID $($target.ProcessId)"
        }
        catch {
            Write-Warning "Unable to resume PID $($target.ProcessId): $($_.Exception.Message)"
        }
    }
    Write-Host "Component inverse V2 dataset generation resumed."
    exit 0
}

if (-not $PythonExe) {
    $pythonCandidates = @(
        (Join-Path $root ".venv\Scripts\python.exe"),
        "C:\Users\20991\anaconda3\python.exe"
    )
    $PythonExe = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $PythonExe) {
        $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            $PythonExe = $pythonCommand.Source
        }
    }
}
if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable was not found. Pass -PythonExe with a valid python.exe path."
}

if ($TargetSeeds -eq 150) {
    $manifest = "configs\sweeps\component_inverse_v2_stage1.yaml"
    $catalogBase = "configs\sweeps\component_inverse_v2_stage1_catalog"
    $expectedCases = 1308
}
else {
    $manifest = "configs\sweeps\component_inverse_v2.yaml"
    $catalogBase = "configs\sweeps\component_inverse_v2_catalog"
    $expectedCases = 2808
}

Write-Host "Preparing Component Inverse V2 manifest: seeds=$TargetSeeds, cases=$expectedCases"
& $PythonExe "utils\generate_component_inverse_v2.py" `
    "--target-seed-count" $TargetSeeds `
    "--output" $manifest `
    "--catalog-json" "$catalogBase.json" `
    "--catalog-csv" "$catalogBase.csv"
if ($LASTEXITCODE -ne 0) {
    throw "V2 manifest generation failed with exit code $LASTEXITCODE"
}
if ($PrepareOnly) {
    Write-Host "Preparation completed. No simulation process was started."
    exit 0
}

$controllerDir = "configs\trials\component_inverse_v2"
New-Item -ItemType Directory -Path $controllerDir -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $controllerDir "controller_${TargetSeeds}seeds_$stamp.out.log"
$errLog = Join-Path $controllerDir "controller_${TargetSeeds}seeds_$stamp.err.log"

$arguments = @(
    "-u",
    "utils\run_param_sweep.py",
    "--python-exe", $PythonExe,
    "--manifest", $manifest,
    "--build-first",
    "--skip-completed",
    "--stop-on-error",
    "--workers", $Workers,
    "--threads-per-worker", $ThreadsPerWorker
)
$env:PYTHONUNBUFFERED = "1"
$process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $arguments `
    -WorkingDirectory $root `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 3
if ($process.HasExited) {
    Write-Host "Controller exited early with code $($process.ExitCode)."
    if (Test-Path -LiteralPath $errLog) {
        Get-Content -LiteralPath $errLog -Encoding UTF8 -Tail 30
    }
    exit $process.ExitCode
}

Write-Host "Component inverse V2 generation started."
Write-Host "Controller PID:       $($process.Id)"
Write-Host "Target:               $TargetSeeds seeds / $expectedCases cases"
Write-Host "Parallel workers:     $Workers"
Write-Host "Threads per worker:   $ThreadsPerWorker"
Write-Host "Completed results will be skipped automatically."
Write-Host "Output log:           $outLog"
Write-Host "Error log:            $errLog"
