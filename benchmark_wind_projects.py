import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import fetch_tmy
import sced_fetcher
from utils import power_curves

def calculate_metrics(actual, modeled):
    """Calculate R, MBE, and RMSE between actual and modeled series."""
    # Ensure they are the same length and aligned
    combined = pd.DataFrame({'actual': actual, 'modeled': modeled}).dropna()
    if combined.empty:
        return {'R': 0, 'MBE': 0, 'RMSE': 0}
        
    actual = combined['actual']
    modeled = combined['modeled']
    
    # --- FILTER: Exclude "Offline" periods ---
    # User concern: "Offline actual project skewing numbers"
    # If Actual is ~0 but Model predicts significant generation (>5 MW), assume outage/maintenance.
    # Exclude these from R/Bias calc as they reflect operational status, not model accuracy.
    
    threshold_mw = 5.0 
    # Mask: Keep only points where (Actual > 0) OR (Model is also low)
    # i.e. Throw away points where (Actual == 0 AND Model > 5)
    
    valid_mask = ~((actual < 0.5) & (modeled > threshold_mw))
    
    actual_filtered = actual[valid_mask]
    modeled_filtered = modeled[valid_mask]
    
    if len(actual_filtered) < 10:
        return {'R': 0, 'MBE': 0, 'RMSE': 0} 
    
    actual = actual_filtered
    modeled = modeled_filtered
    
    # Correlation
    correlation = actual.corr(modeled)
    
    # Mean Bias Error (MBE) - average difference
    mbe = (modeled - actual).mean()
    
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(((modeled - actual)**2).mean())
    
    return {
        'R': correlation,
        'MBE': mbe,
        'RMSE': rmse,
        'Capacity Factor (Actual)': actual.mean() / actual.max() if actual.max() > 0 else 0,
        'Capacity Factor (Modeled)': modeled.mean() / modeled.max() if modeled.max() > 0 else 0
    }

def run_benchmark():
    # 1. Load Assets
    with open('ercot_assets.json', 'r') as f:
        assets = json.load(f)
        
    # Load all wind projects from the registry
    projects = [name for name, meta in assets.items() if meta.get('tech') == 'Wind']
    # Skip problematic resources
    projects = [p for p in projects if p not in ["Monte Cristo 1 Wind", "Monte Cristo Wind"]]
    print(f"Benchmarking confirmed wind projects: {projects}")
    
    # Range for benchmark (Q3 2025 - Summer Peak)
    start_date = "2025-07-01"
    end_date = "2025-09-30"
    # Actually Jan 2026 - 60 days = Nov 2025. So 2024 is fully available.
    
    results = []
    
    for p_name in projects:
        meta = assets.get(p_name)
        if not meta:
            print(f"Skipping {p_name}: no metadata found.")
            continue
            
        print(f"\nBenchmarking {p_name}...")
        r_id = meta['resource_name']
        lat, lon = meta['lat'], meta['lon']
        capacity = meta['capacity_mw']
        hub = meta['hub']
        
        # Actual Hub Specs
        actual_hub_h = meta.get('hub_height_m', 80)
        actual_manuf = meta.get('turbine_manuf', 'GENERIC')
        actual_model = meta.get('turbine_model', 'GENERIC')
        actual_rotor = meta.get('rotor_diameter_m')
        
        # Get derived turbine type
        actual_tech_type = power_curves.get_curve_for_specs(actual_manuf, actual_model, actual_rotor)
        
        # A. Fetch Actual Gen
        print(f"  Fetching actual generation for {r_id}...")
        df_actual = sced_fetcher.get_asset_period_data(r_id, start_date, end_date)
        if df_actual.empty:
            print(f"  ! No actual data found for {r_id} in period.")
            continue
            
        # Ensure UTC and aligned. sced_fetcher returns UTC.
        df_actual = df_actual.set_index('Time')['Actual_MW']
        
        # B. Model 1: Baseline (80m, Generic Curve)
        print(f"  Running Baseline Model (80m, Generic)...")
        prof_baseline = fetch_tmy.get_profile_for_year(
            year=2025, 
            tech="Wind", 
            capacity_mw=capacity, 
            lat=lat, lon=lon, 
            hub_height=80, 
            turbine_type="GENERIC"
        )
        
        # C. Model 2: Advanced (Actual Hub, Actual Curve)
        print(f"  Running Advanced Model ({actual_hub_h}m, {actual_tech_type})...")
        prof_advanced = fetch_tmy.get_profile_for_year(
            year=2025, 
            tech="Wind", 
            capacity_mw=capacity, 
            lat=lat, lon=lon, 
            hub_height=actual_hub_h, 
            turbine_type=actual_tech_type
        )
        
        # D. Align index for comparison
        # Model profiles have a specific year index, actual might start at a specific day.
        # Ensure both are aligned to UTC.
        baseline_aligned = prof_baseline.reindex(df_actual.index).fillna(0)
        advanced_aligned = prof_advanced.reindex(df_actual.index).fillna(0)
        
        # E. Calculate Metrics
        metrics_baseline = calculate_metrics(df_actual, baseline_aligned)
        metrics_advanced = calculate_metrics(df_actual, advanced_aligned)
        
        results.append({
            'Project': p_name,
            'Model': 'Baseline (80m, Generic)',
            'R': metrics_baseline['R'],
            'MBE (MW)': metrics_baseline['MBE'],
            'RMSE (MW)': metrics_baseline['RMSE']
        })
        results.append({
            'Project': p_name,
            'Model': f"Advanced ({actual_hub_h}m, {actual_tech_type})",
            'R': metrics_advanced['R'],
            'MBE (MW)': metrics_advanced['MBE'],
            'RMSE (MW)': metrics_advanced['RMSE']
        })
        
    # Print Summary Table
    df_res = pd.DataFrame(results)
    print("\n" + "="*80)
    print(f"BENCHMARK RESULTS ({start_date} to {end_date})")
    print("="*80)
    print(df_res.to_string(index=False))
    print("\n" + "="*80)
    
    # Save to JSON for app.py
    output_path = 'benchmark_results_wind.json'
    df_res.to_json(output_path, orient='records', indent=4)
    print(f"Results saved to {output_path}")
    
if __name__ == "__main__":
    run_benchmark()
