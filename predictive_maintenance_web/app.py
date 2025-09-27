#!/usr/bin/env python3
"""
EcoPulse Energy Management & Predictive Maintenance Dashboard
Renewable Energy Monitoring with Predictive Analytics
"""

import sys
import os
import logging
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

# Disable Flask's default request logging for cleaner output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app.logger.disabled = True
log.disabled = True

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
            'hydrogen': {'current': 15.6, 'capacity': 50, 'efficiency': 92.1},
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
            'C': 'hydrogen',   # Meter C -> Hydrogen generators
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
        
        # Map meters to energy systems with more diverse maintenance categories
        meter_systems = {
            'A': {'name': 'Solar Panel Array', 'type': 'Solar'},
            'B': {'name': 'Wind Turbine System', 'type': 'Wind'}, 
            'C': {'name': 'Hydrogen Generator Unit', 'type': 'Hydrogen'},
            'D': {'name': 'Gas Processing Plant', 'type': 'Oil & Gas'}
        }
        
        # Additional maintenance categories
        additional_systems = [
            {'name': 'Battery Storage System', 'type': 'Storage', 'meter_ref': 'A'},
            {'name': 'Grid Connection Infrastructure', 'type': 'Grid', 'meter_ref': 'B'},
            {'name': 'HVAC Cooling System', 'type': 'HVAC', 'meter_ref': 'C'},
            {'name': 'Electrical Distribution Panel', 'type': 'Electrical', 'meter_ref': 'D'}
        ]
        
        item_id = 1
        
        # Generate primary maintenance items from UFD data
        for meter_id, data in self.ufd_data.items():
            try:
                # Analyze recent sensor data
                recent_data = data.tail(10)
                avg_health = recent_data['health_state'].mean()
                health_trend = self.energy_sources[meter_systems[meter_id]['type'].lower().replace(' & ', '_')]['degradation_trend']
                
                # Get key sensor readings and generate maintenance item
                system_info = meter_systems[meter_id]
                maintenance_item = self._create_maintenance_item(item_id, system_info, meter_id, avg_health, health_trend, recent_data)
                maintenance_items.append(maintenance_item)
                item_id += 1
                
            except Exception as e:
                print(f"Error generating maintenance for meter {meter_id}: {e}")
        
        # Generate additional maintenance items for supporting systems
        for system in additional_systems:
            try:
                # Use related meter's health data as baseline
                related_data = self.ufd_data[system['meter_ref']].tail(10)
                avg_health = related_data['health_state'].mean() + np.random.uniform(-0.5, 0.5)  # Add variation
                avg_health = max(1.0, min(4.0, avg_health))  # Keep in valid range
                
                health_trend = np.random.uniform(-0.02, 0.02)  # Random trend for supporting systems
                
                maintenance_item = self._create_maintenance_item(item_id, system, system['meter_ref'], avg_health, health_trend, related_data)
                maintenance_items.append(maintenance_item)
                item_id += 1
                
            except Exception as e:
                print(f"Error generating maintenance for {system['name']}: {e}")
        
        self.maintenance_items = maintenance_items
        print(f"Generated {len(maintenance_items)} maintenance items from UFD data")
        
        # Generate usage history based on UFD data patterns
        self.generate_usage_history()
    
    def _create_maintenance_item(self, item_id, system_info, meter_ref, avg_health, health_trend, sensor_data):
        """Helper method to create a comprehensive maintenance item with detailed information"""
        
        # Determine priority based on health state
        if avg_health >= 3.5:
            urgency = 'Critical'
            cost_multiplier = 2.0
            days_multiplier = 1.5
        elif avg_health >= 2.5:
            urgency = 'High'
            cost_multiplier = 1.2
            days_multiplier = 1.0
        elif avg_health >= 1.5:
            urgency = 'Medium'
            cost_multiplier = 0.8
            days_multiplier = 0.7
        else:
            urgency = 'Low'
            cost_multiplier = 0.5
            days_multiplier = 0.5
        
        # Static costs by system type and priority (no longer random)
        static_costs = {
            'Solar': {
                'Critical': 28500, 'High': 18200, 'Medium': 12800, 'Low': 7500
            },
            'Wind': {
                'Critical': 45000, 'High': 28750, 'Medium': 19200, 'Low': 12500
            },
            'Hydrogen': {
                'Critical': 35600, 'High': 21800, 'Medium': 15400, 'Low': 9200
            },
            'Oil & Gas': {
                'Critical': 42300, 'High': 26400, 'Medium': 17600, 'Low': 11800
            },
            'Storage': {
                'Critical': 22800, 'High': 14400, 'Medium': 9600, 'Low': 6000
            },
            'Grid': {
                'Critical': 16800, 'High': 9600, 'Medium': 6400, 'Low': 4200
            },
            'HVAC': {
                'Critical': 12500, 'High': 7500, 'Medium': 4800, 'Low': 2500
            },
            'Electrical': {
                'Critical': 15600, 'High': 9200, 'Medium': 5600, 'Low': 3400
            }
        }
        
        # Base timeframes by system type
        system_configs = {
            'Solar': {'days': 3}, 'Wind': {'days': 5}, 'Hydrogen': {'days': 4},
            'Oil & Gas': {'days': 6}, 'Storage': {'days': 2}, 'Grid': {'days': 1},
            'HVAC': {'days': 1}, 'Electrical': {'days': 2}
        }
        
        config = system_configs.get(system_info['type'], {'days': 3})
        cost = static_costs.get(system_info['type'], {}).get(urgency, 10000)
        
        # Generate comprehensive maintenance details
        details = self._generate_comprehensive_maintenance_details(
            system_info['type'], urgency, avg_health, sensor_data, meter_ref
        )
        
        # Calculate due date based on urgency
        due_days = np.random.randint(1, int(config['days'] * days_multiplier) + 5)
        due_date = (datetime.now() + timedelta(days=due_days)).strftime('%Y-%m-%d')
        
        return {
            'id': item_id,
            'equipment': system_info['name'],
            'issue': details['primary_issue'],
            'priority': urgency,
            'cost': cost,  # Static cost
            'due_date': due_date,
            'estimated_days': int(config['days'] * days_multiplier),
            'energy_impact': round((avg_health - 1) * 5, 1),
            'category': system_info['type'],
            'status': self.determine_status(avg_health, health_trend),
            
            # Comprehensive details for dropdown
            'detailed_info': {
                'root_cause': details['root_cause'],
                'symptoms': details['symptoms'],
                'detection_method': details['detection_method'],
                'maintenance_required': details['maintenance_required'],
                'parts_needed': details['parts_needed'],
                'tools_required': details['tools_required'],
                'safety_precautions': details['safety_precautions'],
                'estimated_downtime': details['estimated_downtime'],
                'environmental_impact': details['environmental_impact']
            },
            
            # Contact information
            'contacts': details['contacts'],
            
            # Technical data for graphs
            'sensor_data': {
                'health_history': self._generate_health_timeline(avg_health, sensor_data),
                'detection_location': details['detection_location'],
                'failure_progression': details['failure_progression']
            },
            
            'meter_data': {
                'meter_id': meter_ref,
                'health_state': round(avg_health, 2),
                'trend': health_trend,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
        }
    
    def _generate_comprehensive_maintenance_details(self, system_type, priority, health_state, sensor_data, meter_id):
        """Generate comprehensive maintenance details with root cause analysis"""
        
        # System-specific maintenance database
        maintenance_db = {
            'Solar': {
                'Critical': {
                    'primary_issue': 'Critical inverter failure with power output loss',
                    'root_cause': 'Excessive heat buildup due to cooling fan failure, causing power electronics degradation. UFD sensor detected anomalous thermal patterns and electrical signature changes indicating imminent component failure.',
                    'symptoms': ['Power output dropped by 35%', 'Inverter temperature >85°C', 'Unusual electrical noise', 'DC/AC conversion efficiency <92%'],
                    'detection_method': f'UFD Meter {meter_id} ultrasonic flow analysis detected irregular electrical flow patterns and thermal anomalies',
                    'maintenance_required': [
                        '1. Complete inverter replacement with upgraded cooling system',
                        '2. Thermal management system overhaul',
                        '3. DC combiner box inspection and cleaning',
                        '4. Power cable integrity testing and replacement if needed',
                        '5. Grounding system verification and repair'
                    ],
                    'parts_needed': [
                        'String inverter (SMA Sunny Boy 7.7kW) - $2,800',
                        'DC combiner box with fuses - $450',
                        'MC4 connectors (20 pcs) - $180',
                        'Copper grounding wire (50ft) - $95',
                        'Thermal interface compound - $35'
                    ],
                    'contacts': {
                        'primary': {'name': 'Marcus Chen', 'role': 'Senior Solar Technician', 'phone': '+1-555-0147', 'email': 'marcus.chen@ecopulse.com'},
                        'backup': {'name': 'Sarah Rodriguez', 'role': 'Solar Systems Engineer', 'phone': '+1-555-0148', 'email': 'sarah.rodriguez@ecopulse.com'},
                        'supervisor': {'name': 'David Kim', 'role': 'Renewable Energy Manager', 'phone': '+1-555-0149', 'email': 'david.kim@ecopulse.com'}
                    },
                    'detection_location': {'lat': 25.7617, 'lng': -80.1918, 'zone': 'Solar Array Block C'},
                    'failure_progression': [85, 78, 65, 52, 38, 28, 15],  # Performance degradation over time
                },
                'High': {
                    'primary_issue': 'Panel soiling and micro-crack development',
                    'root_cause': 'Accumulated dust and debris reducing light absorption by 18%. Recent weather patterns and improper cleaning have led to micro-crack formation in 12 panels.',
                    'symptoms': ['Reduced power output in affected string', 'Hot spot formation', 'Visible soiling patterns', 'Electrical mismatch'],
                    'detection_method': f'UFD Meter {meter_id} detected flow irregularities in DC current patterns',
                    'maintenance_required': [
                        '1. Professional panel cleaning with deionized water',
                        '2. Thermal imaging inspection for hot spots',
                        '3. Electrical testing of affected panels',
                        '4. Panel replacement if micro-cracks exceed 15% area'
                    ],
                    'parts_needed': [
                        'Replacement solar panels (if needed) - $4,200',
                        'Professional cleaning solution - $85',
                        'Panel mounting hardware - $150'
                    ],
                    'contacts': {
                        'primary': {'name': 'Lisa Park', 'role': 'Solar Maintenance Tech', 'phone': '+1-555-0150', 'email': 'lisa.park@ecopulse.com'},
                        'backup': {'name': 'Marcus Chen', 'role': 'Senior Solar Technician', 'phone': '+1-555-0147', 'email': 'marcus.chen@ecopulse.com'}
                    },
                    'detection_location': {'lat': 25.7620, 'lng': -80.1915, 'zone': 'Solar Array Block A'},
                    'failure_progression': [95, 88, 82, 76, 71, 68, 65]
                }
            },
            'Wind': {
                'Critical': {
                    'primary_issue': 'Main gearbox bearing failure with metal debris contamination',
                    'root_cause': 'Excessive vibration and inadequate lubrication caused catastrophic bearing failure. Metal particles in oil sample indicate advanced wear. Wind patterns and turbulence data from UFD sensors show correlation with failure progression.',
                    'symptoms': ['Abnormal vibration levels >8mm/s', 'Oil temperature >75°C', 'Metal particles in oil', 'Unusual noise during operation'],
                    'detection_method': f'UFD Meter {meter_id} ultrasonic sensors detected bearing signature changes and oil flow anomalies',
                    'maintenance_required': [
                        '1. Complete gearbox replacement or rebuild',
                        '2. Oil system flush and replacement',
                        '3. Vibration monitoring system calibration',
                        '4. Bearing housing inspection and repair',
                        '5. Coupling alignment verification'
                    ],
                    'parts_needed': [
                        'Main gearbox assembly - $15,200',
                        'Synthetic gear oil (200L) - $1,800',
                        'Bearing sets (various sizes) - $3,400',
                        'Oil filtration system - $950',
                        'Vibration sensors - $650'
                    ],
                    'contacts': {
                        'primary': {'name': 'Robert Johnson', 'role': 'Wind Turbine Specialist', 'phone': '+1-555-0151', 'email': 'robert.johnson@ecopulse.com'},
                        'backup': {'name': 'Amanda Torres', 'role': 'Mechanical Engineer', 'phone': '+1-555-0152', 'email': 'amanda.torres@ecopulse.com'},
                        'supervisor': {'name': 'James Wilson', 'role': 'Wind Operations Manager', 'phone': '+1-555-0153', 'email': 'james.wilson@ecopulse.com'}
                    },
                    'detection_location': {'lat': 25.7625, 'lng': -80.1925, 'zone': 'Wind Turbine WT-03'},
                    'failure_progression': [92, 85, 74, 58, 41, 25, 12]
                }
            },
            'Hydrogen': {
                'Critical': {
                    'primary_issue': 'Electrolyzer stack membrane degradation with gas crossover',
                    'root_cause': 'Proton exchange membrane (PEM) degradation due to impure water feedstock and excessive operating temperatures. Gas purity analysis shows hydrogen-oxygen crossover indicating membrane failure.',
                    'symptoms': ['Hydrogen purity <99.95%', 'Increased power consumption', 'Stack temperature >80°C', 'Pressure differential anomalies'],
                    'detection_method': f'UFD Meter {meter_id} detected pressure wave anomalies and flow irregularities in the electrolysis process',
                    'maintenance_required': [
                        '1. Complete membrane electrode assembly (MEA) replacement',
                        '2. Water treatment system overhaul',
                        '3. Gas separation system inspection',
                        '4. Pressure vessel integrity testing',
                        '5. Safety system recalibration'
                    ],
                    'parts_needed': [
                        'PEM electrolyzer stack - $12,500',
                        'Water purification filters - $850',
                        'Gas separation membranes - $1,200',
                        'Pressure regulators - $650',
                        'Safety relief valves - $450'
                    ],
                    'contacts': {
                        'primary': {'name': 'Dr. Elena Vasquez', 'role': 'Hydrogen Systems Engineer', 'phone': '+1-555-0154', 'email': 'elena.vasquez@ecopulse.com'},
                        'backup': {'name': 'Michael Chang', 'role': 'Process Control Specialist', 'phone': '+1-555-0155', 'email': 'michael.chang@ecopulse.com'}
                    },
                    'detection_location': {'lat': 25.7612, 'lng': -80.1922, 'zone': 'Hydrogen Generation Unit H2-01'},
                    'failure_progression': [88, 79, 68, 54, 39, 26, 18]
                }
            },
            'Oil & Gas': {
                'Critical': {
                    'primary_issue': 'Pipeline pressure vessel fatigue cracking with potential leak risk',
                    'root_cause': 'Cyclic pressure loading and corrosion under insulation (CUI) have created fatigue cracks in the main process pipeline. Ultrasonic testing reveals crack propagation approaching critical length.',
                    'symptoms': ['Pressure drop of 15 PSI', 'Unusual vibration patterns', 'Temperature anomalies', 'Slight hydrocarbon odor'],
                    'detection_method': f'UFD Meter {meter_id} ultrasonic sensors detected wall thickness variations and flow disturbances',
                    'maintenance_required': [
                        '1. Pipeline section replacement with upgraded materials',
                        '2. Comprehensive ultrasonic testing of adjacent sections',
                        '3. Insulation system replacement',
                        '4. Corrosion protection system upgrade',
                        '5. Pressure testing and recertification'
                    ],
                    'parts_needed': [
                        'Pipeline sections (carbon steel) - $8,900',
                        'Flanges and fittings - $1,500',
                        'Corrosion-resistant coating - $750',
                        'Thermal insulation - $650',
                        'Pressure test equipment rental - $400'
                    ],
                    'contacts': {
                        'primary': {'name': 'Captain Jake Morrison', 'role': 'Pipeline Integrity Specialist', 'phone': '+1-555-0156', 'email': 'jake.morrison@ecopulse.com'},
                        'backup': {'name': 'Rachel Thompson', 'role': 'Process Safety Engineer', 'phone': '+1-555-0157', 'email': 'rachel.thompson@ecopulse.com'}
                    },
                    'detection_location': {'lat': 25.7608, 'lng': -80.1928, 'zone': 'Process Unit PU-04'},
                    'failure_progression': [90, 82, 71, 57, 42, 29, 16]
                }
            }
        }
        
        # Get appropriate details or create generic ones
        category_data = maintenance_db.get(system_type, {})
        priority_data = category_data.get(priority)
        
        if not priority_data:
            # Generic fallback for missing combinations
            priority_data = {
                'primary_issue': f'{system_type} system requires {priority.lower()} priority maintenance',
                'root_cause': f'System health degradation detected through UFD sensor analysis indicating performance issues',
                'symptoms': ['Performance degradation', 'Unusual sensor readings', 'Efficiency loss'],
                'detection_method': f'UFD Meter {meter_id} detected anomalous patterns',
                'maintenance_required': ['System inspection required', 'Performance testing', 'Component replacement if needed'],
                'parts_needed': ['To be determined after inspection'],
                'contacts': {
                    'primary': {'name': 'Operations Team', 'role': 'Maintenance Coordinator', 'phone': '+1-555-0100', 'email': 'operations@ecopulse.com'}
                },
                'detection_location': {'lat': 25.7615, 'lng': -80.1920, 'zone': f'{system_type} System'},
                'failure_progression': [80, 75, 68, 60, 52, 45, 38]
            }
        
        # Add common fields
        priority_data.update({
            'tools_required': ['Multimeter', 'Thermal camera', 'Ultrasonic tester', 'Safety equipment'],
            'safety_precautions': ['Lockout/Tagout procedures', 'Personal protective equipment', 'Gas detection if applicable', 'Hot work permits'],
            'estimated_downtime': f'{2 if priority == "Critical" else 1} days',
            'environmental_impact': 'Temporary reduction in renewable energy output during maintenance'
        })
        
        return priority_data
    
    def _generate_health_timeline(self, current_health, sensor_data):
        """Generate health timeline data for graphing"""
        # Create a 30-day health timeline
        timeline = []
        base_health = current_health
        
        for i in range(30):
            date = (datetime.now() - timedelta(days=29-i)).strftime('%Y-%m-%d')
            
            # Simulate health degradation over time with some randomness
            health_variation = np.random.uniform(-0.3, 0.1)  # Slight downward trend
            base_health = max(1.0, min(4.0, base_health + health_variation))
            
            timeline.append({
                'date': date,
                'health_score': round(base_health, 2),
                'performance': round((5 - base_health) / 4 * 100, 1)  # Convert to performance %
            })
        
        return timeline
        
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
                'hydrogen_output': round(140 * self.energy_sources['hydrogen']['efficiency']/100, 1),
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
            'Hydrogen': ['Hydrogen Engineer', 'Fuel Cell Systems Team'],
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
        # Sort by priority and energy impact
        priority_values = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        
        sorted_items = sorted(
            self.maintenance_items, 
            key=lambda x: (priority_values.get(x.get('priority', 'Low'), 0), x.get('energy_impact', 0)), 
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

@app.route('/current-status-fixed')
def current_status_fixed():
    """Fixed current status page for debugging"""
    return render_template('current_status_fixed.html')

@app.route('/future-projection')
def future_projection():
    """Future projection page with predictive analytics"""
    return render_template('future_projection.html')

# API Routes
@app.route('/api/home_data')
def get_home_data():
    """Get data for home page dashboard"""
    try:
        # Ensure all required data exists
        if not hasattr(ems, 'current_status'):
            ems.current_status = {
                'power_percentage': 87.5,
                'clean_percentage': 72.3,
                'net_zero_percentage': 68.9
            }
        
        if not hasattr(ems, 'maintenance_items') or not ems.maintenance_items:
            ems.generate_realistic_maintenance_data()
        
        # Get critical maintenance items
        critical_maintenance = ems.get_critical_maintenance_items(3)
        
        return jsonify({
            'status': 'success',
            'power_percentage': ems.current_status.get('power_percentage', 87.5),
            'clean_percentage': ems.current_status.get('clean_percentage', 72.3),
            'net_zero_percentage': ems.current_status.get('net_zero_percentage', 68.9),
            'critical_maintenance': critical_maintenance
        })
    except Exception as e:
        print(f"Error in get_home_data: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': str(e),
            'power_percentage': 87.5,
            'clean_percentage': 72.3,
            'net_zero_percentage': 68.9,
            'critical_maintenance': []
        })

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
                    {'A': 'solar', 'B': 'wind', 'C': 'hydrogen', 'D': 'oil_gas'}[meter_id]
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