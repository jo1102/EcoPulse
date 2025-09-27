#!/bin/bash
# EcoPulse Energy Management System Launcher

echo "🌱 Starting EcoPulse Energy Management System..."
echo "⚡ Renewable Energy Monitoring & Predictive Maintenance"
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null
then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "predictive_maintenance_web/app.py" ]; then
    echo "❌ Please run this script from the EcoPulse root directory"
    exit 1
fi

# Navigate to the web app directory
cd predictive_maintenance_web

# Check if requirements are installed
echo "📦 Checking dependencies..."
python -c "import flask, pandas, numpy, sklearn, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing required packages..."
    pip install -r requirements.txt
fi

echo ""
echo "🚀 Launching EcoPulse Dashboard..."
echo "🏠 Home Dashboard: http://localhost:5000"
echo "📊 Current Status: http://localhost:5000/current-status"
echo "🔮 Future Projections: http://localhost:5000/future-projection"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask application
python app.py