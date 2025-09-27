# EcoPulse Energy Management & Predictive Maintenance System

🌱 **Renewable Energy Monitoring with Advanced Predictive Analytics**

EcoPulse is a comprehensive energy management dashboard that combines real-time renewable energy monitoring with AI-powered predictive maintenance capabilities. Built using machine learning models trained on actual UFD (Ultra Flow Diagnostics) sensor data.

## 🚀 Features

### Energy Management
- **Real-time Energy Monitoring** - Live tracking of power generation across multiple sources
- **Multi-Source Integration** - Solar, Wind, Hydrogen, and Oil & Gas systems
- **Interactive Dashboards** - Beautiful, responsive web interface with live data updates
- **Energy Source Distribution** - Dynamic pie charts showing current energy mix
- **Historical Analysis** - 30-day usage trends and performance metrics

### Predictive Maintenance
- **AI-Powered Predictions** - Machine learning models trained on real UFD sensor data
- **Equipment Health Monitoring** - Real-time health state tracking (1-4 scale)
- **Maintenance Scheduling** - Automated maintenance item generation based on sensor data
- **Priority Classification** - High/Medium/Low priority maintenance alerts
- **Performance Degradation Tracking** - Trend analysis for early failure detection

### Technical Highlights
- **Real UFD Data Integration** - Uses actual sensor data from 4 different meters (A, B, C, D)
- **Random Forest ML Models** - Trained with 88-100% accuracy on real sensor data
- **Live Data Simulation** - Dynamic updates every 5-10 seconds
- **Responsive Design** - Bootstrap 5 with custom Aramco-inspired styling
- **Interactive Visualizations** - Plotly.js charts with real-time updates

## 📊 Dashboard Pages

1. **Home Dashboard** (`/`)
   - Overview of system status
   - Key performance indicators
   - Critical maintenance alerts

2. **Current Status** (`/current-status`)
   - Real-time energy level gauge
   - Energy sources distribution chart
   - Active maintenance items
   - Usage history trends

3. **Future Projections** (`/future-projection`)
   - Predictive maintenance forecasts
   - Performance projections
   - Cost estimates and recommendations

## 🔧 Technical Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Visualization**: Plotly.js
- **UI Framework**: Bootstrap 5
- **Icons**: Font Awesome 6
- **Machine Learning**: scikit-learn (Random Forest)
- **Data Processing**: pandas, numpy
- **Real Data**: UFD sensor datasets with 37 features per meter

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/jo1102/EcoPulse.git
cd EcoPulse

# Install dependencies
cd predictive_maintenance_web
pip install -r requirements.txt

# Start the application
python app.py
```

### Access the Dashboard
- **Local Development**: http://localhost:5000
- **Home**: http://localhost:5000/
- **Current Status**: http://localhost:5000/current-status
- **Future Projections**: http://localhost:5000/future-projection

## 📁 Project Structure

```
EcoPulse/
├── predictive_maintenance_web/          # Main web application
│   ├── app.py                          # Flask application & ML models
│   ├── requirements.txt                # Python dependencies
│   ├── templates/                      # HTML templates
│   │   ├── home.html                   # Home dashboard
│   │   ├── current_status.html         # Current status page
│   │   ├── current_status_fixed.html   # Improved version
│   │   └── future_projection.html      # Future projections
│   └── static/                         # Static assets (CSS, JS, images)
├── predictive-maintenance/              # Core ML & data processing
│   ├── datasets/                       # Real UFD sensor datasets
│   │   ├── ufd/                        # UFD meter data (A, B, C, D)
│   │   ├── cmapss/                     # NASA turbofan data
│   │   ├── gfd/                        # Gear fault diagnosis data
│   │   └── ...                         # Other datasets
│   └── notebooks/                      # Jupyter analysis notebooks
├── output/                             # Generated reports & analysis
└── README.md                           # This file
```

## 🤖 Machine Learning Models

### UFD Sensor Integration
- **4 Meter Systems**: A (Solar), B (Wind), C (Hydrogen), D (Oil & Gas)
- **37 Sensor Features**: Including flatness_ratio, symmetry, crossflow, etc.
- **Health States**: 1-4 scale (1=Excellent, 4=Critical)
- **Model Performance**: 88-100% accuracy across all meters

### Predictive Maintenance Features
- **Equipment Health Monitoring**: Real-time degradation tracking
- **Failure Prediction**: Early warning system based on sensor trends
- **Maintenance Scheduling**: Automated task generation
- **Cost Optimization**: Maintenance cost estimates and scheduling

## 🌍 Energy Sources Monitored

1. **Solar Panels** (UFD Meter A)
   - Current: ~35.2 MW
   - Efficiency: Variable based on health state
   - Health tracking via UFD sensor data

2. **Wind Turbines** (UFD Meter B)
   - Current: ~28.7 MW
   - Efficiency: Variable based on health state
   - Performance degradation monitoring

3. **Hydrogen Generators** (UFD Meter C)
   - Current: ~15.6 MW
   - High efficiency ratings
   - Advanced fault detection

4. **Oil & Gas Systems** (UFD Meter D)
   - Current: ~20.5 MW
   - Traditional backup systems
   - Integrated with renewable sources

## 📈 Key Metrics Tracked

- **Power Generation**: Real-time MW output per source
- **Clean Energy Percentage**: % of renewable vs traditional
- **Net Zero Score**: Progress toward carbon neutrality
- **Equipment Efficiency**: Performance degradation over time
- **Maintenance Costs**: Predictive vs reactive maintenance savings
- **System Health**: Overall infrastructure condition

## 🔍 Advanced Features

### Live Data Simulation
- Real-time data updates every 5-10 seconds
- Simulated sensor variations and trends
- Live/historical data toggle

### Interactive Visualizations
- Energy gauge with color-coded performance
- Dynamic pie charts for energy distribution
- Time-series plots for historical trends
- Maintenance timeline visualizations

### Responsive Design
- Mobile-friendly interface
- Aramco-inspired color scheme
- Professional dashboard styling
- Accessible UI components

## 🚀 Future Enhancements

- [ ] IoT sensor integration for real-time data streaming
- [ ] Advanced ML models (LSTM, Transformer) for better predictions
- [ ] Mobile app development
- [ ] API integration with external energy management systems
- [ ] Enhanced reporting and analytics
- [ ] Multi-site deployment capabilities

## 📝 License

This project is part of the predictive maintenance research initiative and uses open-source datasets for educational and research purposes.

## 🤝 Contributing

This is a research and development project. For collaboration or questions, please open an issue or submit a pull request.

---

**Built with ❤️ for sustainable energy management and predictive maintenance**

⚡ Powered by Real UFD Sensor Data | 🤖 AI-Enhanced Predictions | 🌱 Carbon-Neutral Focus