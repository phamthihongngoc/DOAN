@echo off
cd /d G:\DOAN2\training
if not exist G:\DOAN2\training\results_lstm_100epoch mkdir G:\DOAN2\training\results_lstm_100epoch
echo Training LSTM for 100 epochs
echo Results: G:\DOAN2\training\results_lstm_100epoch
echo.
call G:\DOAN2\.venv\Scripts\activate.bat
python -u G:\DOAN2\training\train.py --model lstm --epochs 100 --results-dir G:\DOAN2\training\results_lstm_100epoch
echo.
echo Training finished.
