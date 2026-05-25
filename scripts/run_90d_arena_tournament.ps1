# 90-day SmartAPI backfill + TradingView-style arena + tournament + comparison
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$Symbol = if ($env:FORTUNA_SYMBOL) { $env:FORTUNA_SYMBOL } else { "CROMPTON.NS" }
$Days = 90

function Write-Step($msg) {
    Write-Host ""
    Write-Host "========== $msg ==========" -ForegroundColor Cyan
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "ERROR: .venv missing. Run: uv sync" -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH = "src"
$env:FORTUNA_CONFIG = "configs/intraday.yaml"
$env:FORTUNA_USE_GPU = "1"
$env:FORTUNA_LOW_MEMORY = "1"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path

Write-Step "STEP 1: Backfill $Symbol 5m ($Days days)"
& $py scripts\smartapi_backfill.py --symbol $Symbol --timeframe 5m --days $Days --force
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Step "STEP 2: Strategy comparison (full history, TV-style metrics)"
& $py scripts\run_strategy_compare.py --symbol $Symbol --timeframe 5m --days $Days
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Step "STEP 3: Arena (param search, full series)"
& $py scripts\run_arena.py `
    --symbol $Symbol `
    --timeframe 5m `
    --days $Days `
    --mode full `
    --window-bars 78 `
    --warmup-bars 78 `
    --max-candidates 27 `
    --time-budget 120 `
    --param-grid strategies/grids/ema_grid.json `
    --output logs/arena_90d
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Step "STEP 4: Strategy Tester reports (TradingView-style)"
& $py scripts\run_strategy_tester_report.py --symbol $Symbol --timeframe 5m --days $Days --init-cash 100000 --no-charts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Step "STEP 5: Paper competition (equal capital per strategy)"
& $py scripts\run_paper_competition.py --symbol $Symbol --timeframe 5m --days $Days --init-cash 100000
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Step "STEP 6: Tournament (per-bar winner, bar_step=10 for speed)"
& $py scripts\run_tournament.py `
    --symbol $Symbol `
    --timeframe 5m `
    --days $Days `
    --window-bars 78 `
    --max-candidates 50 `
    --time-budget 90 `
    --param-grid strategies/grids/ema_grid.json `
    --bar-step 10
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "========== 90-DAY PIPELINE COMPLETE ==========" -ForegroundColor Green
Write-Host "  Compare: logs/compare/${Symbol}_5m/"
Write-Host "  Arena:   logs/arena_90d/"
Write-Host "  Tournament: logs/tournament/"
