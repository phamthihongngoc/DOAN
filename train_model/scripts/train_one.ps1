param(
    [ValidateSet("random_forest", "cnn", "lstm", "transformer")]
    [string]$Model = "cnn"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "G:\DOAN2\train_model"

if ($Model -eq "random_forest") {
    & "G:\DOAN2\.venv\Scripts\python.exe" -u -m src.train_random_forest --config "configs\default.json"
} else {
    & "G:\DOAN2\.venv\Scripts\python.exe" -u -m src.train_deep --model $Model --config "configs\default.json"
}
