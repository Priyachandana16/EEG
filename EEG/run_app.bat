@echo off
title NeuroTrust EEG Stress Detection Dashboard
echo ===================================================
echo Starting EEG Stress Detection Streamlit Application
echo ===================================================

cd /d "%~dp0"

IF NOT EXIST ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Installing requirements...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) ELSE (
    call .venv\Scripts\activate.bat
)

echo.
echo Launching Streamlit...
python -m streamlit run app.py
pause
