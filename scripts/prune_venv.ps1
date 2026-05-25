# Prune optional / heavy packages from .venv (PowerShell, repo root)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..



if (-not (Test-Path ".venv\Scripts\python.exe")) {

    Write-Host "No .venv found." -ForegroundColor Red

    exit 1

}



$pip = ".\.venv\Scripts\python.exe -m pip"

$remove = @(

    "pytest-cov", "coverage",

    "jupyter", "jupyterlab", "notebook", "ipykernel", "ipython",

    "jupyter-client", "jupyter-core", "nbconvert", "nbformat"

)



Write-Host "Removing optional packages (ignore errors if not installed)..." -ForegroundColor Cyan

foreach ($pkg in $remove) {

    Invoke-Expression "$pip uninstall -y $pkg 2>`$null" | Out-Null

}



Write-Host "To drop vectorbt stack (tournament-only): uv sync (without --group vbt) after updating pyproject.toml" -ForegroundColor Yellow

Write-Host "Recommended lean install: uv sync && uv sync --group dev" -ForegroundColor Green

