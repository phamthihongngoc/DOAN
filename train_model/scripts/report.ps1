$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "G:\DOAN2\train_model"

& "G:\DOAN2\.venv\Scripts\python.exe" -u -m src.report --config "configs\default.json"
