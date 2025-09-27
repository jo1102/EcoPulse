@echo off
echo ========================================
echo  Predictive Maintenance Dashboard
echo ========================================
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo.
echo Installing required packages...
python -m pip install --user -r requirements.txt

echo.
echo Starting the dashboard...
echo.
echo ========================================
echo  Dashboard will be available at:
echo  http://localhost:5000
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause