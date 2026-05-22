$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "G:\DOAN2\training_50hz_clean\src"

& "G:\DOAN2\.venv\Scripts\python.exe" -u "train.py" `
  --model transformer `
  --epochs 100 `
  --results-dir "G:\DOAN2\training_50hz_clean\checkpoints\transformer_100epoch"
