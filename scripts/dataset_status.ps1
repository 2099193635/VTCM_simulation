param(
    [ValidateSet(150, 300)]
    [int]$TargetSeeds = 150,
    [ValidateRange(1, 16)]
    [int]$RecentWorkers = 3
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$expectedCases = if ($TargetSeeds -eq 150) { 1308 } else { 2808 }
$resultRoot = "results\component_inverse_v2"
$logRoot = "configs\trials\component_inverse_v2\logs"

$completed = @(
    Get-ChildItem -LiteralPath $resultRoot `
        -Recurse `
        -Filter "simulation_result.npz" `
        -File `
        -ErrorAction SilentlyContinue
).Count
$percent = if ($expectedCases -gt 0) {
    [math]::Min(100.0, [math]::Round(100.0 * $completed / $expectedCases, 2))
}
else { 0.0 }

try {
    $targets = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
        $_.Name -eq "python.exe" -and $_.CommandLine -like "*component_inverse_v2*"
    })
}
catch {
    Write-Warning "Unable to query active Python command lines: $($_.Exception.Message)"
    $targets = @()
}
$controllers = @($targets | Where-Object { $_.CommandLine -like "*run_param_sweep.py*" })
$workers = @($targets | Where-Object { $_.CommandLine -like "*generate_main.py*" })

Write-Host "Component inverse V2 status"
Write-Host "Target seeds:          $TargetSeeds"
Write-Host "Completed simulations: $completed / $expectedCases ($percent%)"
Write-Host "Active controllers:    $($controllers.Count)"
Write-Host "Active workers:        $($workers.Count)"

$logs = @(
    Get-ChildItem -LiteralPath $logRoot `
        -Filter "*.err.log" `
        -File `
        -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First $RecentWorkers
)

if ($logs.Count -eq 0) {
    Write-Host "No case logs are available yet."
    exit 0
}

Write-Host ""
Write-Host "Most recent worker logs:"
foreach ($log in $logs) {
    $progressLine = Get-Content -LiteralPath $log.FullName -Encoding UTF8 -Tail 80 |
        Where-Object { $_ -match "(\d+)%.*?(\d+)/(\d+).*?\[([0-9:]+)<" } |
        Select-Object -Last 1
    if ($progressLine -and $progressLine -match "(\d+)%.*?(\d+)/(\d+).*?\[([0-9:]+)<") {
        Write-Host "- $($log.BaseName): $($Matches[1])%, step $($Matches[2])/$($Matches[3]), elapsed $($Matches[4])"
    }
    else {
        Write-Host "- $($log.BaseName): initializing, completed, or progress unavailable"
    }
}

