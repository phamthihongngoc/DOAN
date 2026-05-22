@echo off
cd /d G:\DOAN2\train_model
if not exist G:\DOAN2\train_model\logs mkdir G:\DOAN2\train_model\logs
set LOG=G:\DOAN2\train_model\logs\run_all.log
G:\DOAN2\.venv\Scripts\python.exe -u -m src.run_all --config configs\default.json > %LOG% 2>&1
echo Log: %LOG%
