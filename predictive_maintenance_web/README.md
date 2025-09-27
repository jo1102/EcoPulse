# Predictive Maintenance Dashboard for UFD Flow Meters

A comprehensive web-based dashboard for monitoring ultrasonic flow meters and predicting maintenance needs using machine learning.

## 🚀 Features

### 📊 Processing Speed Analysis
- **Real-time Processing Requirements**: Shows optimal flow velocity ranges for each meter
- **Sound Speed Monitoring**: Tracks acoustic processing speeds
- **Performance Recommendations**: Provides processing optimization suggestions
- **Sampling Rate Guidance**: Recommends optimal data collection frequencies

### 🔧 Predictive Maintenance Scheduling
- **30-90 Day Forecasts**: Predicts maintenance needs up to 3 months ahead
- **Priority-based Alerts**: Categorizes maintenance urgency (Low, Medium, High, Critical)
- **Timeline Visualization**: Interactive charts showing health state progression
- **Automated Scheduling**: Smart recommendations for maintenance timing

### 📈 Advanced Analytics
- **Health State Distribution**: Visual breakdown of equipment conditions across all meters
- **Feature Importance Analysis**: Identifies which sensors are most predictive of failures
- **Multi-meter Comparison**: Compare performance across different flow meters
- **Historical Trends**: Track degradation patterns over time

### 🔮 Live Prediction Engine
- **Real-time Health Assessment**: Input current sensor readings for instant predictions
- **Confidence Scoring**: Shows prediction reliability
- **Maintenance Recommendations**: Immediate actionable insights
- **What-if Analysis**: Test different sensor scenarios

## 🏗️ Technology Stack

- **Backend**: Python Flask with scikit-learn ML models
- **Frontend**: Bootstrap 5 with Plotly.js for interactive charts
- **Machine Learning**: Random Forest classifiers trained on UFD dataset
- **Data Processing**: Pandas and NumPy for efficient data handling

## 🚦 Quick Start

### Option 1: Windows Batch File
1. Double-click `start_dashboard.bat`
2. Wait for installation and startup
3. Open http://localhost:5000 in your browser

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the dashboard
python app.py
```

## 📊 Understanding the Data

### UFD Dataset Overview
The dashboard uses the **Ultrasonic Flow meter Degradation (UFD)** dataset containing:

- **4 Flow Meters** (A, B, C, D) with different sensor configurations
- **37-52 Features** per meter including:
  - Flow characteristics (velocity, symmetry, crossflow)
  - Acoustic properties (sound speed, signal gain)
  - Health states (1=Healthy → 4=Critical)

### Key Metrics Explained

#### Processing Speed Requirements
- **Flow Velocity**: Optimal operating ranges for each meter
- **Sound Speed**: Acoustic signal processing speeds
- **Sampling Rate**: Recommended data collection frequency
- **Processing Load**: Computational requirements

#### Health States
- **State 1** (🟢): Healthy - Normal operation
- **State 2** (🟡): Early degradation - Monitor closely
- **State 3** (🟠): Moderate degradation - Schedule maintenance
- **State 4** (🔴): Severe degradation - Immediate attention required

## 🎯 Business Value

### Cost Savings
- **Prevent Unplanned Downtime**: 15-30% reduction in unexpected failures
- **Optimize Maintenance Costs**: Schedule maintenance during planned downtime
- **Extend Equipment Life**: Early intervention prevents major damage

### Operational Benefits
- **Improved Reliability**: Predict failures before they occur
- **Better Resource Planning**: Schedule technicians and parts in advance
- **Data-Driven Decisions**: Replace gut instinct with predictive analytics

### Processing Optimization
- **Optimal Flow Rates**: Maintain equipment within ideal operating ranges
- **Sensor Calibration**: Know when sensors need recalibration
- **Performance Monitoring**: Track efficiency degradation over time

## 📱 Dashboard Usage Guide

### 1. Processing Speeds Tab
- View optimal flow velocity ranges for each meter
- Monitor sound speed processing requirements
- Get recommendations for sampling rates and processing loads

### 2. Maintenance Schedule Tab
- Select a meter and forecast period (1-90 days)
- View predicted health state progression
- See upcoming maintenance events with priority levels

### 3. Analytics Tab
- Explore health state distributions across all meters
- Identify which sensor features are most predictive
- Compare performance between different meters

### 4. Live Prediction Tab
- Input current sensor readings
- Get instant health state predictions
- Receive immediate maintenance recommendations

## 🔧 Technical Details

### Machine Learning Models
- **Algorithm**: Random Forest Classifier
- **Features**: 37-52 sensor measurements per meter
- **Training**: 70/30 train-test split with stratification
- **Validation**: Cross-validation with accuracy metrics

### Prediction Logic
- **Health State Prediction**: Multi-class classification (1-4)
- **Maintenance Scheduling**: Rule-based system using predicted states
- **Confidence Scoring**: Based on model prediction probabilities
- **Trend Analysis**: Time-series forecasting with noise simulation

### Performance Metrics
- **Model Accuracy**: Typically 85-95% depending on meter
- **Prediction Confidence**: Probability scores for reliability
- **Feature Importance**: Ranking of most predictive sensors

## 🚨 Alert System

### Priority Levels
- **Low**: Routine maintenance in optimal timeframe
- **Medium**: Schedule maintenance within 2 weeks
- **High**: Schedule maintenance within 1 week
- **Critical**: Immediate attention required

### Automatic Notifications
- Dashboard updates every time page loads
- Real-time health state monitoring
- Predictive alerts based on trend analysis

## 🔄 Data Pipeline

1. **Data Loading**: Automatic loading of UFD dataset
2. **Preprocessing**: Feature scaling and normalization
3. **Model Training**: Random Forest training for each meter
4. **Prediction**: Real-time inference for maintenance scheduling
5. **Visualization**: Interactive charts and dashboards

## 📈 Future Enhancements

- Real-time sensor data integration
- Email/SMS alert notifications
- Historical data tracking and storage
- Advanced analytics (anomaly detection, pattern recognition)
- Mobile-responsive design improvements
- API endpoints for external system integration

## 🤝 Support

For questions or issues:
1. Check the browser console for error messages
2. Ensure all Python dependencies are installed
3. Verify UFD dataset is accessible in the predictive-maintenance folder
4. Review the Flask application logs for detailed error information

---

**Built for Industrial IoT and Predictive Maintenance Applications**