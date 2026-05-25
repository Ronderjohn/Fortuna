# Verbose pipeline: SmartAPI backfill -> arena quick (no SilentlyContinue)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

function Write-Step($msg) {
    Write-Host ""
    Write-Host "========== $msg ==========" -ForegroundColor Cyan
}

Write-Step "STEP 0: Repo and shell"
Write-Host "  CWD: $(Get-Location)"
Write-Host "  Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

Write-Step "STEP 1: Python / uv processes BEFORE"
try {
    Get-Process -Name python*, pytest*, uv | Format-Table Id, ProcessName, CPU, @{
        N = "RAM_MB"; E = { [math]::Round($_.WorkingSet64 / 1MB, 1) }
    } -AutoSize
} catch {
    Write-Host "  (no python/pytest/uv processes running)"
}

Write-Step "STEP 2: Stop python / pytest / uv"
try {
    $toStop = Get-Process -Name python*, pytest*, uv
    foreach ($p in $toStop) {
        Write-Host "  Stopping PID $($p.Id) $($p.ProcessName)"
        Stop-Process -Id $p.Id -Force
    }
    Start-Sleep -Seconds 2
} catch {
    Write-Host "  Nothing to stop."
}

Write-Step "STEP 3: Environment"
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "ERROR: .venv missing. Run: uv sync" -ForegroundColor Red
    exit 1
}
$env:PYTHONPATH = "src"
$env:FORTUNA_CONFIG = "configs/intraday.yaml"
$env:FORTUNA_USE_GPU = "1"
$env:FORTUNA_LOW_MEMORY = "1"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
Write-Host "  PYTHONPATH=$env:PYTHONPATH"
Write-Host "  FORTUNA_CONFIG=$env:FORTUNA_CONFIG"
Write-Host "  FORTUNA_USE_GPU=$env:FORTUNA_USE_GPU"
Write-Host "  FORTUNA_LOW_MEMORY=$env:FORTUNA_LOW_MEMORY"
Write-Host "  Python: $py"

Write-Step "STEP 4: SmartAPI backfill (historical -> Parquet)"
& $py scripts\smartapi_backfill.py --symbol RELIANCE.NS --timeframe 5m --days 7 --force
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: smartapi_backfill failed with exit $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Step "STEP 5: Arena quick (3 strategies, param search)"
& $py scripts\run_arena_quick.py
$arenaExit = $LASTEXITCODE
if ($arenaExit -ne 0) {
    Write-Host "ERROR: run_arena_quick failed with exit $arenaExit" -ForegroundColor Red
    exit $arenaExit
}

Write-Step "STEP 6: Python processes AFTER"
try {
    Get-Process -Name python* | Format-Table Id, ProcessName, CPU, @{
        N = "RAM_MB"; E = { [math]::Round($_.WorkingSet64 / 1MB, 1) }
    } -AutoSize
} catch {
    Write-Host "  (no python processes)"
}

Write-Host ""
Write-Host "========== PIPELINE COMPLETE ==========" -ForegroundColor Green
