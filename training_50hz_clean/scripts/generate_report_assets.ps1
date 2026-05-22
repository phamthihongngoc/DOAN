$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "G:\DOAN2\training_50hz_clean\src"

& "G:\DOAN2\.venv\Scripts\python.exe" -u "generate_report_assets_50hz.py"
