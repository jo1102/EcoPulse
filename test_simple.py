import sys
import os
sys.path.append(r'c:\Users\jonhe\Downloads\Ecopulse\predictive-maintenance')

try:
    import datasets
    import pandas as pd
    import numpy as np
    
    print("=== UFD Dataset Test ===")
    
    # Test loading data for each meter
    for meter_id in ["A", "B", "C", "D"]:
        print(f"\n--- Meter {meter_id} ---")
        try:
            data = datasets.ufd.load_data(meter_id=meter_id)
            print(f"Shape: {data.shape}")
            print(f"Columns: {len(data.columns)}")
            print(f"Health states: {sorted(data['health_state'].unique())}")
            print(f"Health state counts:")
            print(data['health_state'].value_counts().to_dict())
        except Exception as e:
            print(f"Error loading meter {meter_id}: {e}")
    
    # Create a simple summary for Meter A
    print(f"\n=== Detailed Summary for Meter A ===")
    data_a = datasets.ufd.load_data(meter_id="A")
    
    print("First 5 rows:")
    print(data_a.head())
    
    print("\nBasic statistics:")
    print(data_a.describe())
    
    print("\nData types:")
    print(data_a.dtypes.value_counts())
    
    print("\nMissing values:")
    print(data_a.isnull().sum().sum())
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have installed the requirements from requirement.txt")
except Exception as e:
    print(f"Error: {e}")