@echo off
chcp 65001 >nul
set "TARGET_SEEDS=%~1"
set "WORKERS=%~2"
set "THREADS_PER_WORKER=%~3"
if not defined TARGET_SEEDS set "TARGET_SEEDS=150"
if not defined WORKERS set "WORKERS=3"
if not defined THREADS_PER_WORKER set "THREADS_PER_WORKER=1"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\dataset_start.ps1" -TargetSeeds %TARGET_SEEDS% -Workers %WORKERS% -ThreadsPerWorker %THREADS_PER_WORKER%
pause

