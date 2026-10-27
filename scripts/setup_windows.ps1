# VisionGuard Lite - Windows PowerShell Setup Script
# Creates a reproducible CPU-first Python virtual environment.

param(
    [string]$PythonCommand = "py -3.11",
    [switch]$Dev = $false
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   VisionGuard Lite: Windows CPU Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Resolve Python executable
$pythonExe = $null
try {
    $pythonExe = (Get-Command ((-split $PythonCommand)[0]) -ErrorAction SilentlyContinue).Source
    Write-Host "[+] Using Python command: $PythonCommand" -ForegroundColor Green
} catch {
    Write-Host "[*] Specified Python command not found, falling back to 'python'..." -ForegroundColor Yellow
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonCommand = "python"
    } else {
        Write-Error "Python was not found on PATH. Please install Python 3.10 or 3.11 from python.org."
        exit 1
    }
}

# 2. Project Root Directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot
Write-Host "[+] Working in repository root: $projectRoot" -ForegroundColor Green

# 3. Create Virtual Environment
$venvPath = Join-Path $projectRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "[+] Creating virtual environment at .venv..." -ForegroundColor Cyan
    & ($PythonCommand -split " ") -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
} else {
    Write-Host "[+] Virtual environment already exists at .venv." -ForegroundColor Green
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

# 4. Upgrade pip and build tools
Write-Host "[+] Upgrading pip and wheel..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip wheel setuptools

# 5. Install CPU-first dependencies
Write-Host "[+] Installing CPU-first dependencies from requirements-local.txt..." -ForegroundColor Cyan
& $venvPip install -r (Join-Path $projectRoot "requirements-local.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install base dependencies."
    exit 1
}

if ($Dev) {
    Write-Host "[+] Installing development dependencies from requirements-dev.txt..." -ForegroundColor Cyan
    & $venvPip install -r (Join-Path $projectRoot "requirements-dev.txt")
}

# 6. Setup .env configuration
$envFile = Join-Path $projectRoot ".env"
$envExample = Join-Path $projectRoot ".env.example"
if (-not (Test-Path $envFile) -and (Test-Path $envExample)) {
    Write-Host "[+] Copying .env.example to .env..." -ForegroundColor Cyan
    Copy-Item $envExample $envFile
}

Write-Host "`n[SUCCESS] Setup complete!" -ForegroundColor Green
Write-Host "To launch the Gradio demo: .\scripts\run_local.ps1" -ForegroundColor Yellow
Write-Host "To run test suite:         .\scripts\run_tests.ps1" -ForegroundColor Yellow
