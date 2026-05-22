$ErrorActionPreference = "Stop"

Set-Location -LiteralPath "G:\DOAN2\training"

$resultsDir = "G:\DOAN2\training\results_cnn_100epoch"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

Write-Host "Training 1D-CNN for 100 epochs"
Write-Host "Results: $resultsDir"
Write-Host ""

& "..\.venv\Scripts\python.exe" -u "train.py" --model cnn --epochs 100 --results-dir $resultsDir

Write-Host ""
Write-Host "Training finished. Press Enter to close this terminal."
Read-Host | Out-Null
