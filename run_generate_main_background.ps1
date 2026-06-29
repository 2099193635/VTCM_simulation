param(
    [string]$PythonExe = "python",
    [string]$ScriptName = "generate_main.py"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptPath = Join-Path $ProjectRoot $ScriptName
$LogDir = Join-Path $ProjectRoot "logs"
$OutLog = Join-Path $LogDir "generate_main.out.log"
$ErrLog = Join-Path $LogDir "generate_main.err.log"
$PidFile = Join-Path $LogDir "generate_main.pid"

if (-not (Test-Path $ScriptPath)) {
    throw "Script not found: $ScriptPath"
}

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList "`"$ScriptPath`"" `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

$process.Id | Set-Content -Path $PidFile -Encoding ASCII

Write-Host "Started $ScriptName in background."
Write-Host "PID: $($process.Id)"
Write-Host "stdout: $OutLog"
Write-Host "stderr/progress: $ErrLog"
Write-Host "pid file: $PidFile"
Write-Host ""
Write-Host "Watch progress:"
Write-Host "  Get-Content `"$ErrLog`" -Wait"
Write-Host ""
Write-Host "Stop process:"
Write-Host "  Stop-Process -Id $($process.Id)"
