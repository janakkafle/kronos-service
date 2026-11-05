$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
Set-Location $rootDir
$env:PYTHONPATH = "$rootDir;$scriptDir;" + $env:PYTHONPATH
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting Kronos-base Financial API on http://127.0.0.1:8001" -ForegroundColor Green
Write-Host "Working Directory: $rootDir" -ForegroundColor DarkGray
Write-Host "========================================================" -ForegroundColor Cyan
python -m uvicorn kronos.app:app --host 127.0.0.1 --port 8001 --reload
