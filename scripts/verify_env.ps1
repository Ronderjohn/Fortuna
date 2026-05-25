# Verify Fortuna Python environment (run in PowerShell from repo root)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..



Write-Host "Fortuna environment check" -ForegroundColor Cyan



if (-not (Test-Path ".venv\Scripts\python.exe")) {

    Write-Host "ERROR: .venv not found. Run: uv sync && uv sync --group dev" -ForegroundColor Red

    exit 1

}



$py = ".\.venv\Scripts\python.exe"

$env:PYTHONPATH = "src"



& $py -c @"

import sys

import time



def timed_import(name):

    t0 = time.perf_counter()

    __import__(name)

    ms = (time.perf_counter() - t0) * 1000

    print(f'  {name}: {ms:.0f} ms')



print('Python', sys.version.split()[0])

for pkg in ('numpy', 'pandas', 'numba', 'fortuna', 'pytest'):

    timed_import(pkg)

try:

    timed_import('vectorbt')

except Exception:

    print('  vectorbt: not installed (optional: uv sync --group vbt)')

"@



if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }



Write-Host "`nRunning fast tests..." -ForegroundColor Cyan

& $py -m pytest tests/ -q -m "not slow and not integration"

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }



Write-Host "`nOK" -ForegroundColor Green

