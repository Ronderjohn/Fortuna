# Full intraday suite: backfill -> institutional backtest -> paper competition -> strategy tester reports
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$Symbol = if ($env:FORTUNA_SYMBOL) { $env:FORTUNA_SYMBOL } else { "ICICIBANK.NS" }
$Days = if ($env:FORTUNA_DAYS) { [int]$env:FORTUNA_DAYS } else { 30 }
$MinDays = if ($env:FORTUNA_MIN_DAYS) { [int]$env:FORTUNA_MIN_DAYS } else { 20 }

function Write-Step($msg) {
    Write-Host ""
    Write-Host "========== $msg ==========" -ForegroundColor Cyan
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "ERROR: .venv missing" -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH = "src"
$env:FORTUNA_CONFIG = "configs/intraday.yaml"
$env:FORTUNA_USE_GPU = "1"
$env:FORTUNA_LOW_MEMORY = "1"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path

Write-Step "STEP 1: Backfill $Symbol ($Days days)"
& $py scripts\smartapi_backfill.py --symbol $Symbol --timeframe 5m --days $Days --force
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Step "STEP 2: Institutional backtest (train/val/OOS + filters)"
& $py scripts\run_institutional_backtest.py --symbol $Symbol --days $Days --min-days $MinDays --min-trades 10
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Step "STEP 3: Paper competition (equal capital per strategy)"
& $py scripts\run_paper_competition.py --symbol $Symbol --days $Days --init-cash 100000
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Step "STEP 4: Strategy Tester reports (all intraday strategies)"
& $py scripts\run_strategy_tester_report.py --symbol $Symbol --days $Days
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "========== INTRADAY SUITE COMPLETE ==========" -ForegroundColor Green
Write-Host "  Institutional: logs/institutional/"
Write-Host "  Paper:         logs/paper_competition/"
Write-Host "  TV reports:    logs/strategy_tester/  (matplotlib PNG charts per strategy)"
Write-Host "  Paper charts:  logs/paper_competition/<run>/charts/"
Write-Host "  Strategies:    strategies/intraday/ + generated/ + builtin/"
