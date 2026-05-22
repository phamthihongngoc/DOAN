$ErrorActionPreference = "Stop"

Set-Location -LiteralPath "G:\DOAN2\train_model"

$logDir = "G:\DOAN2\train_model\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "run_all_$stamp.log"

Write-Host "Running full training pipeline"
Write-Host "Log: $logPath"

& "G:\DOAN2\.venv\Scripts\python.exe" -u -m src.run_all --config "configs\default.json" 2>&1 |
    Tee-Object -FilePath $logPath
