@echo off
chcp 65001 >nul
set "TARGET_SEEDS=%~1"
if not defined TARGET_SEEDS set "TARGET_SEEDS=150"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\dataset_status.ps1" -TargetSeeds %TARGET_SEEDS%
pause

