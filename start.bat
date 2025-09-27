@echo off
REM EcoPulse Energy Management System Launcher for Windows

echo 🌱 Starting EcoPulse Energy Management System...
echo ⚡ Renewable Energy Monitoring ^& Predictive Maintenance
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "predictive_maintenance_web\app.py" (
    echo ❌ Please run this script from the EcoPulse root directory
    pause
    exit /b 1
)

REM Navigate to the web app directory
cd predictive_maintenance_web

REM Check if requirements are installed
echo 📦 Checking dependencies...
python -c "import flask, pandas, numpy, sklearn, plotly" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing required packages...
    pip install -r requirements.txt
)

echo.
echo 🚀 Launching EcoPulse Dashboard...
echo 🏠 Home Dashboard: http://localhost:5000
echo 📊 Current Status: http://localhost:5000/current-status
echo 🔮 Future Projections: http://localhost:5000/future-projection
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the Flask application
python app.py

pause