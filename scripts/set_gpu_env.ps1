# Session env for GTX 1650 / 8GB RAM — GPU on, ProcessPool off on Windows
$env:FORTUNA_USE_GPU = "1"
$env:FORTUNA_LOW_MEMORY = "1"
$env:FORTUNA_GPU_MIN_BARS = "32"
$env:FORTUNA_GPU_MIN_SERIES = "2"
$env:FORTUNA_GPU_MEM_FRACTION = "0.75"
$env:FORTUNA_CPU_WORKERS_GPU = "1"
Write-Host "Fortuna GPU env set (USE_GPU=1, LOW_MEMORY=1 for OOM-safe runs)" -ForegroundColor Green
