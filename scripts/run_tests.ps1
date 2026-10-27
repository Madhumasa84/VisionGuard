# VisionGuard Lite - Run Unit & Integration Tests (Windows PowerShell)

param(
    [string]$Target = "tests",
    [switch]$Coverage = $false
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment not detected. Run .\scripts\setup_windows.ps1 first."
    exit 1
}

$env:PYTHONPATH = $projectRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Executing VisionGuard Lite Test Suite" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if ($Coverage) {
    & $venvPython -m pytest $Target -v --cov=src --cov-report=term-missing
} else {
    & $venvPython -m pytest $Target -v
}
