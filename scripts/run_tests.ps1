# Reliable test runner for 8GB Windows — one folder at a time, 90s timeout each
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Get-Process python*, pytest* -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "ERROR: Run uv sync && uv sync --group dev" -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH = "src"
$env:FORTUNA_USE_GPU = "0"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$timeoutSec = 90

$suites = @(
    "tests/test_strategy_schema.py",
    "tests/test_scorer.py",
    "tests/test_risk.py",
    "tests/test_indicators.py",
    "tests/test_compiler.py",
    "tests/backtest",
    "tests/search",
    "tests/tournament",
    "tests/reporting",
    "tests/arena",
    "tests/paper",
    "tests/quant",
    "tests/compute",
    "tests/data"
)

$failed = 0
foreach ($target in $suites) {
    if (-not (Test-Path $target)) { continue }
    Write-Host "`n--- $target ---" -ForegroundColor Cyan
    # Single Arguments string — array form splits "-m not slow..." and pytest treats "slow" as a path
    $argStr = "-m pytest `"$target`" -q --tb=line -m `"not slow and not integration`" -p no:cov -o addopts="
    $proc = Start-Process -FilePath $py -ArgumentList $argStr -NoNewWindow -PassThru
    $done = $proc.WaitForExit($timeoutSec * 1000)
    if (-not $done) {
        Write-Host "TIMEOUT (> ${timeoutSec}s) — killing" -ForegroundColor Red
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        $failed++
        continue
    }
    if ($proc.ExitCode -ne 0) {
        Write-Host "FAIL exit $($proc.ExitCode)" -ForegroundColor Red
        $failed++
    } else {
        Write-Host "OK" -ForegroundColor Green
    }
}

if ($failed -gt 0) {
    Write-Host "`n$failed suite(s) failed or timed out" -ForegroundColor Red
    exit 1
}
Write-Host "`nAll suites passed" -ForegroundColor Green
exit 0
