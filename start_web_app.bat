@echo off
echo ========================================
echo Darts Match Predictor Web App
echo ========================================
echo.

REM Change to the directory where this batch file is located
cd /d "%~dp0"
echo Working directory: %CD%
echo.

echo Starting server...
echo.
echo Open your browser and navigate to:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

python app.py

pause
