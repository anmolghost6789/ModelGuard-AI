@echo off
echo ======================================================================
echo  Launching ModelGuard-AI: AI Model Failure Prediction Dashboard
echo ======================================================================
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Failed to launch Streamlit. Please ensure dependencies are installed:
    echo pip install -r requirements.txt
    pause
)
