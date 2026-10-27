# VisionGuard Lite - Run Local Gradio Interface (Windows PowerShell)

param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 7860,
    [string]$Backend = "onnx" # "onnx" or "torch"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Warning "Virtual environment not detected at .venv. Running .\scripts\setup_windows.ps1 first..."
    & (Join-Path $scriptDir "setup_windows.ps1")
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Starting VisionGuard Lite Web Interface (CPU-First)" -ForegroundColor Cyan
Write-Host "   URL: http://$HostAddress`:$Port" -ForegroundColor Green
Write-Host "   Backend: $Backend" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

$env:PYTHONPATH = $projectRoot
$env:DEFAULT_BACKEND = $Backend

& $venvPython (Join-Path $projectRoot "demo\app.py") --host $HostAddress --port $Port
