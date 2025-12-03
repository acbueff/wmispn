@echo off
REM Launch script for WMISPN GUI (Windows)

echo Starting WMISPN GUI...
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>NUL
if errorlevel 1 (
    echo Streamlit is not installed.
    echo Installing requirements...
    pip install -r requirements.txt
)

REM Launch the GUI
streamlit run wmispn_app.py
