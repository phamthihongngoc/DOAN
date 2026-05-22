$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "G:\DOAN2\training_50hz_clean\src"

& "G:\DOAN2\.venv\Scripts\python.exe" -u "random_forest.py" `
  --results-dir "G:\DOAN2\training_50hz_clean\baselines\random_forest"
