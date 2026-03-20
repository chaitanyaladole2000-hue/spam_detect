@echo off
echo =============================================
echo   SpamShield - Spam Email Detector
echo =============================================

REM First run? Set up venv + install deps
if not exist venv (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    echo [2/3] Installing dependencies...
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

REM Train model if not already trained
if not exist model\artifacts\best_model.pt (
    echo [3/3] Training BiLSTM model (first run only, ~5 min)...
    python model\train.py --data spam.tsv --epochs 20 --batch 32
) else (
    echo [3/3] Model already trained. Skipping.
)

echo.
echo Starting SpamShield at http://localhost:5000
echo Press Ctrl+C to stop.
echo.
python app.py
pause
