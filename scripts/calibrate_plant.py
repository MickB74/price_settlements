import sys
import os
import json
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import eia_fetcher
import fetch_tmy

PARAMS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_static', 'plant_calibration.json')

def load_calibration_db():
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_calibration_db(db):
    os.makedirs(os.path.dirname(PARAMS_FILE), exist_ok=True)
    with open(PARAMS_FILE, 'w') as f:
        json.dump(db, f, indent=2)

def calibrate_plant(plant_name, eia_plant_id, year, tech='Wind', capacity_mw=100, lat=32.0, lon=-100.0, turbine_type='GENERIC'):
    print(f"--- Calibrating {plant_name} (ID: {eia_plant_id}) for {year} ---")
    
    # 1. Fetch EIA Actuals
    print("Fetching EIA data...")
    s_actual_monthly = eia_fetcher.get_plant_generation(eia_plant_id, year)
    
    if s_actual_monthly is None or s_actual_monthly.sum() == 0:
        print("❌ No EIA data found or all zero.")
        return None

    # Filter to only months with data (EIA data lags 2-3 months)
    # Assume 0 really means 0 generation, but if trailing months are missing they might technically be NaN or not present.
    # Our fetcher returns 1-12 with 0 fill. Let's be careful.
    # Valid months: where gen > 0? No, curtailment exists. 
    # Best way: Check max month in raw data if possible. 
    # For now, let's assume if Dec is 0 and Jan is >0, Dec is probably missing data if year is current.
    # Simple heuristic: Use all months where value > 0 (risky if legit 0 gen). 
    # Better: Use all months returned by API.
    
    valid_months = s_actual_monthly[s_actual_monthly > 0].index.tolist()
    if not valid_months:
        print("❌ No non-zero data found.")
        return None
        
    print(f"Found valid EIA data for months: {valid_months}")
    
    # 2. Run Model (Gross)
    print("Running Model calculation (Gross)...")
    # Using efficiency=1.0 to get raw potential
    df_profile = fetch_tmy.get_profile_for_year(
        year=year, 
        tech=tech, 
        lat=lat, 
        lon=lon, 
        capacity_mw=capacity_mw, 
        force_tmy=False, 
        turbine_type=turbine_type,
        efficiency=1.0 
    )
    
    if df_profile is None or df_profile.empty:
        print("❌ Model returned no data.")
        return None
        
    # Aggregate Model to Monthly
    df_profile['month'] = df_profile.index.month
    # Normalize column names: fetch_tmy returns 'Gen_MW'. Sum -> MWh conversion (assuming 15-min or hourly?)
    # fetch_tmy returns 15-min series in 'Gen_MW'. Energy = MW * 0.25
    s_model_energy = df_profile.groupby('month')['Gen_MW'].sum() * 0.25 # MWh
    
    # Align
    df_compare = pd.DataFrame({
        'EIA_MWh': s_actual_monthly,
        'Model_Gross_MWh': s_model_energy
    }).dropna()
    
    # Filter to valid months
    df_compare = df_compare.loc[valid_months]
    
    if df_compare.empty:
        print("❌ No overlapping data.")
        return None
        
    # 3. Optimize Efficiency
    # Simple scalar solver: Minimize sum((EIA - (Model * eff))^2)
    # Analytical solution for least squares through origin: eff = dot(x,y) / dot(x,x)
    # where x = Model, y = EIA
    x = df_compare['Model_Gross_MWh'].values
    y = df_compare['EIA_MWh'].values
    
    best_eff = np.dot(x, y) / np.dot(x, x)
    
    # Bound efficiency reasonable range (e.g., 0.5 to 1.1 - allowing some bias calibration > 1 if under-forecasting)
    best_eff = max(0.5, min(1.2, best_eff))
    
    # Calculate Error metrics
    y_pred = x * best_eff
    
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    mae = np.mean(np.abs(y - y_pred))
    
    print(f"✅ Calibration Complete!")
    print(f"   Optimized Efficiency: {best_eff:.4f} ( {(1-best_eff)*100:.1f}% Losses )")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   MAE:  {mae:.0f} MWh/mo")
    
    # Save results
    db = load_calibration_db()
    db[plant_name] = {
        'eia_plant_id': eia_plant_id,
        'tuned_efficiency': round(best_eff, 4),
        'last_calibrated': datetime.now().strftime('%Y-%m-%d'),
        'calibration_year': year,
        'metrics': {
            'mape': round(mape, 2),
            'mae': round(mae, 0)
        }
    }
    save_calibration_db(db)
    print(f"Saved parameters to {PARAMS_FILE}")
    
    return best_eff

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate plant model vs EIA data")
    parser.add_argument("--name", required=True, help="Plant Name (Registry Key)")
    parser.add_argument("--id", required=True, help="EIA Plant ID")
    parser.add_argument("--year", type=int, default=2024, help="Calibration Year")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--mw", type=float, required=True)
    
    args = parser.parse_args()
    
    calibrate_plant(args.name, args.id, args.year, lat=args.lat, lon=args.lon, capacity_mw=args.mw)
