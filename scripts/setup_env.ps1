param(
    [string]$PythonExe = "py -3.13"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Creating virtual environment (.venv)..."
Invoke-Expression "$PythonExe -m venv .venv"

Write-Host "[2/3] Upgrading pip..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "[3/3] Installing dev dependencies..."
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
