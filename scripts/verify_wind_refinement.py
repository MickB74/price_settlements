import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from utils import power_curves
import sced_fetcher
from datetime import datetime, timedelta

def verify_physics():
    print("\n--- Verifying Physics Models ---")
    test_speeds = np.array([3.0, 9.0, 10.5, 12.0, 25.0])
    
    # Generic Curve
    p_generic = power_curves.get_normalized_power(test_speeds, "GENERIC")
    print(f"Generic Curve (10.5 m/s): {p_generic[2]:.2f} (Expected < 1.0)")
    
    # Vestas V163
    p_v163 = power_curves.get_normalized_power(test_speeds, "VESTAS_V163")
    print(f"Vestas V163 (10.5 m/s): {p_v163[2]:.2f} (Expected == 1.0)")
    
    if p_generic[2] < 0.9 and p_v163[2] == 1.0:
        print("✅ Physics check passed: V163 saturates earlier than Generic.")
    else:
        print("❌ Physics check failed.")

def verify_fetcher():
    print("\n--- Verifying Data Ingestion (Base Point) ---")
    # Fetch random day for Monte Cristo (assuming it has data)
    # We'll use a recent date where we expect data
    test_date = "2024-05-15" 
    
    print(f"Fetching data for MONTECR1_WIND1 on {test_date}...")
    df = sced_fetcher.get_asset_actual_gen("MONTECR1_WIND1", test_date)
    
    if df.empty:
        print("⚠️ No data found (might not be downloaded yet). Skipping fetch verify.")
        return

    print("Columns found:", df.columns.tolist())
    
    if 'Base_Point_MW' in df.columns:
        print(f"✅ 'Base_Point_MW' found! Values: {df['Base_Point_MW'].mean():.2f} MW avg")
    else:
        print("❌ 'Base_Point_MW' missing. Fetcher update failed.")

if __name__ == "__main__":
    verify_physics()
    verify_fetcher()
