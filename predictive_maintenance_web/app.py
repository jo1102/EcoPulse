#!/usr/bin/env python3
"""
EcoPulse Energy Management & Predictive Maintenance Dashboard
Renewable Energy Monitoring with Predictive Analytics
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'predictive-maintenance'))

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.utils
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import datasets
from datetime import datetime, timedelta
import random

app = Flask(__name__)

class EnergyManagementSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.ufd_data = {}
        self.energy_data = {}
        self.maintenance_items = []
        self.usage_history = []
        self.current_status = {
            'power_percentage': 87.5,
            'clean_percentage': 72.3,
            'net_zero_percentage': 68.9
        }
        self.energy_sources = {
            'solar': {'current': 35.2, 'capacity': 100, 'efficiency': 85.1},
            'wind': {'current': 28.7, 'capacity': 100, 'efficiency': 78.3},
            'hydro': {'current': 15.6, 'capacity': 50, 'efficiency': 92.1},
            'oil_gas': {'current': 20.5, 'capacity': 80, 'efficiency': 67.8}
        }
        self.load_ufd_data()
        self.train_predictive_models()
        self.generate_realistic_maintenance_data()
        
    def load_ufd_data(self):
        """Load actual UFD sensor data for predictive maintenance"""
        print("Loading UFD sensor data...")
        
        # Map UFD meters to energy sources
        meter_to_source = {
            'A': 'solar',      # Meter A -> Solar panels
            'B': 'wind',       # Meter B -> Wind turbines  
            'C': 'hydro',      # Meter C -> Hydro generators
            'D': 'oil_gas'     # Meter D -> Gas systems
        }
        
        for meter_id in ['A', 'B', 'C', 'D']:
            try:
                data = datasets.ufd.load_data(meter_id=meter_id)
                self.ufd_data[meter_id] = data
                
                # Calculate current health metrics
                latest_health = data['health_state'].iloc[-10:].mean()  # Average of last 10 readings
                degradation_trend = self.calculate_degradation_trend(data)
                
                # Map to energy source
                energy_source = meter_to_source[meter_id]
                
                # Update energy source efficiency based on health state
                base_efficiency = self.energy_sources[energy_source]['efficiency']
                health_factor = (5 - latest_health) / 4  # Convert health state to efficiency factor
                actual_efficiency = base_efficiency * health_factor
                
                self.energy_sources[energy_source]['efficiency'] = round(actual_efficiency, 1)
                self.energy_sources[energy_source]['health_state'] = round(latest_health, 2)
                self.energy_sources[energy_source]['degradation_trend'] = degradation_trend
                
                print(f"UFD Meter {meter_id} ({energy_source}): Health={latest_health:.2f}, Efficiency={actual_efficiency:.1f}%")
                
            except Exception as e:
                print(f"Error loading UFD meter {meter_id}: {e}")
    
    def calculate_degradation_trend(self, data):
        """Calculate degradation trend from UFD data"""
        health_states = data['health_state'].values
        if len(health_states) < 10:
            return 0.0
            
        # Calculate trend over last 20 readings
        recent_states = health_states[-20:] if len(health_states) >= 20 else health_states
        
        # Simple linear trend calculation
        x = np.arange(len(recent_states))
        if len(x) > 1:
            slope = np.polyfit(x, recent_states, 1)[0]
            return round(slope, 3)
        return 0.0
    
    def train_predictive_models(self):
        """Train ML models on UFD data for maintenance prediction"""
        print("Training predictive maintenance models...")
        
        for meter_id, data in self.ufd_data.items():
            try:
                # Prepare features (exclude health_state)
                X = data.drop('health_state', axis=1)
                y = data['health_state']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train Random Forest model
                model = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42, 
                    max_depth=10
                )
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store model and scaler
                self.models[meter_id] = model
                self.scalers[meter_id] = scaler
                
                print(f"UFD Meter {meter_id} model trained with {accuracy:.1%} accuracy")
                
            except Exception as e:
                print(f"Error training model for meter {meter_id}: {e}")
    
    def generate_realistic_maintenance_data(self):
        """Generate maintenance items based on actual UFD sensor data"""
        
        maintenance_items = []
        
        # Map meters to energy systems
        meter_systems = {
            'A': {'name': 'Solar Panel Array', 'type': 'Solar'},
            'B': {'name': 'Wind Turbine System', 'type': 'Wind'}, 
            'C': {'name': 'Hydro Generator Unit', 'type': 'Hydro'},
            'D': {'name': 'Gas Processing Plant', 'type': 'Oil & Gas'}
        }
        
        item_id = 1
        
        for meter_id, data in self.ufd_data.items():
            try:
                # Analyze recent sensor data
                recent_data = data.tail(10)
                avg_health = recent_data['health_state'].mean()
                health_trend = self.energy_sources[meter_systems[meter_id]['type'].lower().replace(' & ', '_')]['degradation_trend']
                
                # Get key sensor readings
                if meter_id == 'A':  # Solar
                    avg_gain = recent_data[[col for col in data.columns if 'gain' in col]].mean().mean()
                    flow_variance = recent_data[[col for col in data.columns if 'flow_velocity' in col]].var().mean()
                elif meter_id == 'B':  # Wind  
                    signal_strength = recent_data[[col for col in data.columns if 'signal_strength' in col]].mean().mean()
                    turbulence = recent_data[[col for col in data.columns if 'turbulence' in col]].mean().mean()
                elif meter_id == 'C':  # Hydro
                    sound_speed_avg = recent_data[[col for col in data.columns if 'sound_speed' in col]].mean().mean()
                    signal_quality = recent_data[[col for col in data.columns if 'signal_quality' in col]].mean().mean()
                elif meter_id == 'D':  # Gas
                    pressure_variance = recent_data[['profile_factor', 'symmetry']].var().mean()
                    transit_time_avg = recent_data[[col for col in data.columns if 'transit_time' in col]].mean().mean()
                
                # Generate maintenance items based on health state and sensor data
                if avg_health >= 3.5:  # Critical condition
                    urgency = 'Critical'
                    cost_multiplier = 2.0
                    days_multiplier = 1.5
                elif avg_health >= 2.5:  # High priority
                    urgency = 'High'  
                    cost_multiplier = 1.2
                    days_multiplier = 1.0
                elif avg_health >= 1.5:  # Medium priority
                    urgency = 'Medium'
                    cost_multiplier = 0.8
                    days_multiplier = 0.7
                else:  # Low priority
                    urgency = 'Low'
                    cost_multiplier = 0.5
                    days_multiplier = 0.5
                
                # Create specific maintenance items based on meter type and sensor data
                system_info = meter_systems[meter_id]
                base_cost = {'Solar': 15000, 'Wind': 25000, 'Hydro': 18000, 'Oil & Gas': 22000}[system_info['type']]
                base_days = {'Solar': 3, 'Wind': 5, 'Hydro': 4, 'Oil & Gas': 6}[system_info['type']]
                
                # Generate maintenance description based on sensor anomalies
                if meter_id == 'A' and avg_gain > 35.8:  # Solar - High gain indicates sensor issues
                    description = f"Solar panel sensor calibration needed. Gain levels at {avg_gain:.1f}, indicating measurement drift."
                elif meter_id == 'B' and turbulence > 2.0:  # Wind - High turbulence
                    description = f"Wind turbine experiencing high turbulence ({turbulence:.2f}). Blade inspection required."
                elif meter_id == 'C' and sound_speed_avg < 1400:  # Hydro - Low sound speed
                    description = f"Hydro system acoustic anomaly detected. Sound speed at {sound_speed_avg:.0f} m/s."
                elif meter_id == 'D' and pressure_variance > 0.1:  # Gas - Pressure instability
                    description = f"Gas system pressure instability detected. Variance at {pressure_variance:.3f}."
                else:
                    description = f"Routine maintenance based on health state analysis (current: {avg_health:.2f})."
                
                maintenance_item = {
                    'id': item_id,
                    'title': f'{system_info["name"]} - {urgency} Maintenance',
                    'description': description,
                    'urgency': urgency,
                    'cost': int(base_cost * cost_multiplier),
                    'estimated_days': int(base_days * days_multiplier),
                    'energy_impact': round((avg_health - 1) * 5, 1),  # Convert health state to energy impact %
                    'notified': self.get_notification_list(system_info['type'], urgency),
                    'category': system_info['type'],
                    'status': self.determine_status(avg_health, health_trend),
                    'meter_data': {
                        'meter_id': meter_id,
                        'health_state': round(avg_health, 2),
                        'trend': health_trend,
                        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
                    }
                }
                
                maintenance_items.append(maintenance_item)
                item_id += 1
                
            except Exception as e:
                print(f"Error generating maintenance for meter {meter_id}: {e}")
        
        self.maintenance_items = maintenance_items
        print(f"Generated {len(maintenance_items)} maintenance items from UFD data")
        
        # Generate usage history based on UFD data patterns
        self.generate_usage_history()
        
    def generate_usage_history(self):
        """Generate historical usage data based on UFD sensor data"""
        base_date = datetime.now() - timedelta(days=30)
        self.usage_history = []
        
        for i in range(30):
            date = base_date + timedelta(days=i)
            
            # Use UFD health states to influence daily performance
            daily_health_impact = 1.0
            daily_maintenance_events = 0
            
            for meter_id, data in self.ufd_data.items():
                if len(data) > i:
                    # Get health state for this "day" (using UFD sample as proxy)
                    sample_idx = min(i, len(data) - 1)
                    health_state = data.iloc[sample_idx]['health_state']
                    
                    # Convert health state to performance impact
                    health_factor = (5 - health_state) / 4  # 1.0 for health=1, 0.25 for health=4
                    daily_health_impact *= health_factor
                    
                    # Count maintenance events (health state changes)
                    if i > 0 and len(data) > i-1:
                        prev_health = data.iloc[sample_idx-1]['health_state']
                        if abs(health_state - prev_health) > 0.5:
                            daily_maintenance_events += 1
            
            # Apply realistic variations with UFD-influenced patterns
            base_performance = daily_health_impact * 0.9  # Base performance from health
            weather_factor = 0.85 + 0.3 * np.random.random()  # Weather variation
            seasonal_factor = 0.9 + 0.2 * np.sin(i * 0.1)  # Seasonal patterns
            
            daily_data = {
                'date': date.strftime('%Y-%m-%d'),
                'power_generated': round(800 + 250 * base_performance * weather_factor * seasonal_factor, 1),
                'clean_energy': round(60 + 25 * base_performance * seasonal_factor, 1),
                'net_zero_score': round(55 + 30 * base_performance * seasonal_factor, 1),
                'solar_output': round(280 * self.energy_sources['solar']['efficiency']/100 * weather_factor, 1),
                'wind_output': round(230 * self.energy_sources['wind']['efficiency']/100 * weather_factor, 1),
                'hydro_output': round(140 * self.energy_sources['hydro']['efficiency']/100, 1),
                'gas_output': round(180 * self.energy_sources['oil_gas']['efficiency']/100, 1),
                'maintenance_events': daily_maintenance_events,
                'efficiency_drop': round(max(0, (1 - daily_health_impact) * 10), 2)  # Efficiency drop based on health
            }
            self.usage_history.append(daily_data)
        
        print(f"Generated {len(self.usage_history)} days of usage history from UFD data")
        
    def get_notification_list(self, system_type, urgency):
        """Get appropriate notification list based on system and urgency"""
        base_contacts = {
            'Solar': ['Solar Team Lead', 'Renewable Operations'],
            'Wind': ['Wind Technician', 'Turbine Specialist'], 
            'Hydro': ['Hydro Engineer', 'Water Systems Team'],
            'Oil & Gas': ['Gas Operations', 'Safety Officer']
        }
        
        contacts = base_contacts.get(system_type, ['Operations Team'])
        
        if urgency in ['Critical', 'High']:
            contacts.extend(['Plant Manager', 'Emergency Response'])
        if urgency == 'Critical':
            contacts.append('Executive Team')
            
        return contacts
    
    def determine_status(self, health_state, trend):
        """Determine maintenance status based on health and trend"""
        if health_state >= 3.5:
            return 'In Progress' if trend > 0.05 else 'Pending'
        elif health_state >= 2.5:
            return 'Scheduled' if trend > 0 else 'Pending'
        else:
            return 'Completed' if random.random() > 0.7 else 'Scheduled'
        
    def predict_maintenance_schedule_ufd(self, meter_id, days_ahead=30):
        """Use UFD trained models to predict future maintenance needs"""
        if meter_id not in self.models or meter_id not in self.ufd_data:
            return None
            
        data = self.ufd_data[meter_id]
        model = self.models[meter_id]
        scaler = self.scalers[meter_id]
        
        # Use recent data patterns
        recent_data = data.tail(10).drop('health_state', axis=1)
        
        schedule = []
        base_date = datetime.now()
        
        for day in range(1, days_ahead + 1):
            # Simulate sensor drift over time using actual sensor patterns
            noise_factor = 0.01 * (day / days_ahead)
            
            # Add realistic degradation patterns based on UFD data
            degradation_factor = 1 + (day * 0.002)  # Gradual degradation over time
            
            predictions = []
            for _, sample in recent_data.iterrows():
                # Apply degradation and noise to simulate future conditions
                future_sample = sample * degradation_factor + np.random.normal(0, noise_factor, len(sample))
                
                # Predict using trained UFD model
                sample_scaled = scaler.transform([future_sample])
                pred = model.predict(sample_scaled)[0]
                prob = model.predict_proba(sample_scaled)[0]
                predictions.append({'state': pred, 'confidence': max(prob)})
            
            # Determine maintenance need based on UFD model predictions
            avg_state = np.mean([p['state'] for p in predictions])
            max_state = max([p['state'] for p in predictions])
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            
            maintenance_needed = False
            urgency = "Low"
            
            if max_state >= 4:
                maintenance_needed = True
                urgency = "Critical"
            elif max_state >= 3:
                maintenance_needed = True
                urgency = "High"
            elif avg_state > 2.3:
                maintenance_needed = True
                urgency = "Medium"
            
            schedule.append({
                'date': (base_date + timedelta(days=day)).strftime('%Y-%m-%d'),
                'day': day,
                'predicted_state': round(avg_state, 2),
                'max_state': round(max_state, 2),
                'confidence': round(avg_confidence, 3),
                'maintenance_needed': maintenance_needed,
                'urgency': urgency,
                'based_on_ufd': True,
                'meter_id': meter_id
            })
        
        return schedule
    
    def get_critical_maintenance_items(self, limit=5):
        """Get most critical maintenance items"""
        # Sort by urgency and energy impact
        urgency_priority = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        
        sorted_items = sorted(
            self.maintenance_items, 
            key=lambda x: (urgency_priority.get(x['urgency'], 0), x['energy_impact']), 
            reverse=True
        )
        
        return sorted_items[:limit]
    
    def predict_future_performance(self, days_ahead=90):
        """Predict future energy performance based on historical data"""
        if not self.usage_history:
            return []
        
        # Simple trend analysis
        recent_data = self.usage_history[-14:]  # Last 2 weeks
        
        avg_clean_trend = np.mean([d['clean_energy'] for d in recent_data])
        avg_power_trend = np.mean([d['power_generated'] for d in recent_data])
        avg_nz_trend = np.mean([d['net_zero_score'] for d in recent_data])
        
        predictions = []
        base_date = datetime.now()
        
        for day in range(1, days_ahead + 1):
            future_date = base_date + timedelta(days=day)
            
            # Add seasonal and maintenance impact factors
            seasonal_factor = 0.95 + 0.1 * np.sin(day * 0.02)
            maintenance_impact = 1.0
            
            # Check for scheduled maintenance impact
            for item in self.maintenance_items:
                if item['status'] in ['Pending', 'Scheduled'] and day < 30:
                    maintenance_impact += item['energy_impact'] / 100
            
            prediction = {
                'date': future_date.strftime('%Y-%m-%d'),
                'day': day,
                'predicted_power': round(avg_power_trend * seasonal_factor * maintenance_impact, 1),
                'predicted_clean': round(avg_clean_trend * seasonal_factor * maintenance_impact, 1),
                'predicted_net_zero': round(avg_nz_trend * seasonal_factor * maintenance_impact, 1),
                'estimated_cost': round(np.random.uniform(1000, 8000) if day < 60 else 0, 0),
                'recommended_actions': self._get_recommended_actions(day)
            }
            predictions.append(prediction)
        
        return predictions
    
    def _get_recommended_actions(self, day):
        """Get recommended actions for future dates"""
        if day < 7:
            return ['Monitor solar panel efficiency', 'Check wind turbine performance']
        elif day < 30:
            return ['Schedule preventive maintenance', 'Optimize energy storage']
        elif day < 60:
            return ['Plan seasonal adjustments', 'Review capacity expansion']
        else:
            return ['Long-term infrastructure planning', 'Technology upgrades']

# Initialize the energy management system
ems = EnergyManagementSystem()

@app.route('/')
def home():
    """Home page with power, clean%, and net zero percentages"""
    return render_template('home.html')

@app.route('/current-status')
def current_status():
    """Current status page with energy monitoring and maintenance"""
    return render_template('current_status.html')

@app.route('/future-projection')
def future_projection():
    """Future projection page with predictive analytics"""
    return render_template('future_projection.html')

# API Routes
@app.route('/api/home_data')
def get_home_data():
    """Get data for home page dashboard"""
    try:
        return jsonify({
            'status': 'success',
            'power_percentage': ems.current_status['power_percentage'],
            'clean_percentage': ems.current_status['clean_percentage'],
            'net_zero_percentage': ems.current_status['net_zero_percentage'],
            'critical_maintenance': ems.get_critical_maintenance_items(3)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/energy_sources')
def get_energy_sources():
    """Get current energy source data"""
    return jsonify({
        'status': 'success',
        'energy_sources': ems.energy_sources,
        'total_power': ems.current_status['power_percentage']
    })

@app.route('/api/maintenance_items')
def get_maintenance_items():
    """Get all maintenance items"""
    category_filter = request.args.get('category', 'all')
    status_filter = request.args.get('status', 'all')
    
    items = ems.maintenance_items
    
    if category_filter != 'all':
        items = [item for item in items if item['category'].lower() == category_filter.lower()]
    
    if status_filter != 'all':
        items = [item for item in items if item['status'].lower() == status_filter.lower()]
    
    return jsonify({
        'status': 'success',
        'maintenance_items': items,
        'total_cost': sum(item['cost'] for item in items if item['status'] != 'Completed'),
        'categories': list(set(item['category'] for item in ems.maintenance_items))
    })

@app.route('/api/usage_history')
def get_usage_history():
    """Get historical usage data"""
    days = int(request.args.get('days', 30))
    
    history = ems.usage_history[-days:]
    
    return jsonify({
        'status': 'success',
        'history': history,
        'summary': {
            'avg_power': round(np.mean([h['power_generated'] for h in history]), 1),
            'avg_clean': round(np.mean([h['clean_energy'] for h in history]), 1),
            'avg_net_zero': round(np.mean([h['net_zero_score'] for h in history]), 1),
            'total_maintenance_events': sum(h['maintenance_events'] for h in history)
        }
    })

@app.route('/api/future_projections')
def get_future_projections():
    """Get future performance projections"""
    days_ahead = int(request.args.get('days', 90))
    
    projections = ems.predict_future_performance(days_ahead)
    
    return jsonify({
        'status': 'success',
        'projections': projections,
        'total_estimated_cost': sum(p['estimated_cost'] for p in projections),
        'improvement_potential': {
            'power': round(np.mean([p['predicted_power'] for p in projections[:30]]), 1),
            'clean': round(np.mean([p['predicted_clean'] for p in projections[:30]]), 1),
            'net_zero': round(np.mean([p['predicted_net_zero'] for p in projections[:30]]), 1)
        }
    })

@app.route('/api/update_maintenance/<int:item_id>', methods=['POST'])
def update_maintenance(item_id):
    """Update maintenance item status"""
    try:
        data = request.get_json()
        new_status = data.get('status')
        
        for item in ems.maintenance_items:
            if item['id'] == item_id:
                item['status'] = new_status
                if new_status == 'Completed':
                    item['completed_date'] = datetime.now().strftime('%Y-%m-%d')
                break
        
        return jsonify({'status': 'success', 'message': 'Maintenance item updated'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/ufd_data_analysis')
def get_ufd_data_analysis():
    """Get analysis of actual UFD sensor data"""
    try:
        analysis = {
            'ufd_integration': True,
            'meters_analyzed': len(ems.ufd_data),
            'models_trained': len(ems.models),
            'meter_details': {}
        }
        
        for meter_id, data in ems.ufd_data.items():
            recent_samples = data.tail(10)
            
            analysis['meter_details'][meter_id] = {
                'total_samples': len(data),
                'features': len(data.columns) - 1,  # Exclude health_state
                'current_health': round(recent_samples['health_state'].mean(), 2),
                'health_trend': ems.energy_sources[
                    {'A': 'solar', 'B': 'wind', 'C': 'hydro', 'D': 'oil_gas'}[meter_id]
                ]['degradation_trend'],
                'recent_readings': {
                    'avg_flatness_ratio': round(recent_samples.get('flatness_ratio', pd.Series([0])).mean(), 3),
                    'avg_symmetry': round(recent_samples.get('symmetry', pd.Series([0])).mean(), 3),
                    'avg_crossflow': round(recent_samples.get('crossflow', pd.Series([0])).mean(), 3),
                },
                'model_accuracy': 'Trained' if meter_id in ems.models else 'Not Available'
            }
        
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'message': 'Dashboard using real UFD sensor data for predictive maintenance'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/predict_ufd_maintenance/<meter_id>')
def predict_ufd_maintenance(meter_id):
    """Get UFD-based maintenance predictions for a specific meter"""
    try:
        days_ahead = int(request.args.get('days', 30))
        schedule = ems.predict_maintenance_schedule_ufd(meter_id, days_ahead)
        
        if schedule is None:
            return jsonify({'status': 'error', 'message': f'No UFD model available for meter {meter_id}'})
        
        return jsonify({
            'status': 'success',
            'meter_id': meter_id,
            'schedule': schedule,
            'data_source': 'UFD Sensor Data',
            'model_type': 'Random Forest trained on real sensor readings'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("🌱 Starting EcoPulse Energy Management Dashboard...")
    print("⚡ Renewable Energy Monitoring & Predictive Maintenance")
    print("✅ Initial setup complete!")
    
    print("\n🚀 Starting web server...")
    print("🏠 Home Dashboard: http://localhost:5000")
    print("📊 Current Status: http://localhost:5000/current-status") 
    print("� Future Projections: http://localhost:5000/future-projection")
    
    app.run(debug=True, host='0.0.0.0', port=5000)