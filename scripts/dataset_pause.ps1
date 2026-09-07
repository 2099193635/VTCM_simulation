$ErrorActionPreference = "Stop"

try {
    $targets = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
        $_.Name -eq "python.exe" -and $_.CommandLine -like "*component_inverse_v2*"
    })
}
catch {
    throw "Unable to query Component Inverse V2 processes: $($_.Exception.Message)"
}

if ($targets.Count -eq 0) {
    Write-Host "No running Component Inverse V2 dataset task was found."
    exit 0
}

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class DatasetPause {
    [DllImport("ntdll.dll")]
    public static extern int NtSuspendProcess(IntPtr handle);
}
"@ -ErrorAction SilentlyContinue

# Pause workers first, then the controller, so no replacement case is scheduled midway.
$targets |
    Sort-Object { if ($_.CommandLine -like "*generate_main.py*") { 0 } else { 1 } } |
    ForEach-Object {
        try {
            $process = Get-Process -Id $_.ProcessId -ErrorAction Stop
            [void][DatasetPause]::NtSuspendProcess($process.Handle)
            Write-Host "Paused PID $($_.ProcessId)"
        }
        catch {
            Write-Warning "Unable to pause PID $($_.ProcessId): $($_.Exception.Message)"
        }
    }

Write-Host "Component inverse V2 dataset generation paused."
Write-Host "Run dataset_start.bat again to resume all suspended processes."
