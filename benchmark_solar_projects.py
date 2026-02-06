import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import fetch_tmy
import sced_fetcher

def calculate_metrics(actual, modeled):
    """Calculate R, MBE, and RMSE between actual and modeled series."""
    combined = pd.DataFrame({'actual': actual, 'modeled': modeled}).dropna()
    if combined.empty:
        return {'R': 0, 'MBE': 0, 'RMSE': 0}
        
    actual = combined['actual']
    modeled = combined['modeled']
    
    # --- FILTER: Exclude "Offline" periods ---
    # User concern: "Offline actual project skewing numbers"
    # If Actual is ~0 but Model predicts significant generation (>5 MW), assume outage/maintenance.
    # Exclude these from R/Bias calc as they reflect operational status, not model accuracy.
    
    threshold_mw = 5.0 # Assume 5 MW is minimal active generation for utility scale
    # Mask: Keep only points where (Actual > 0) OR (Model is also low)
    # i.e. Throw away points where (Actual == 0 AND Model > 5)
    
    valid_mask = ~((actual < 0.5) & (modeled > threshold_mw))
    
    actual_filtered = actual[valid_mask]
    modeled_filtered = modeled[valid_mask]
    
    if len(actual_filtered) < 10:
        return {'R': 0, 'MBE': 0, 'RMSE': 0} # Too much filtered
    
    actual = actual_filtered
    modeled = modeled_filtered
    
    correlation = actual.corr(modeled)
    mbe = (modeled - actual).mean()
    rmse = np.sqrt(((modeled - actual)**2).mean())
    
    return {
        'R': correlation,
        'MBE': mbe,
        'RMSE': rmse
    }

def run_benchmark():
    # 1. Load Assets
    if not os.path.exists('ercot_assets.json'):
        print("ercot_assets.json not found.")
        return
        
    with open('ercot_assets.json', 'r') as f:
        assets = json.load(f)
        
    # Load solar projects
    projects = [name for name, meta in assets.items() if meta.get('tech') == 'Solar']
    print(f"Benchmarking solar projects: {projects}")
    
    # Range for benchmark (12 Months: Dec 2024 - Nov 2025)
    start_date = "2024-12-01"
    end_date = "2025-11-30"
    
    results = []
    
    for p_name in projects:
        meta = assets.get(p_name)
        r_id = meta['resource_name']
        lat, lon = meta['lat'], meta['lon']
        capacity = meta['capacity_mw']
        
        print(f"\nBenchmarking {p_name}...")
        
        # A. Fetch Actual Gen
        print(f"  Fetching actual generation for {r_id}...")
        df_actual_full = sced_fetcher.get_asset_period_data(r_id, start_date, end_date)
        if df_actual_full.empty:
            print(f"  ! No actual data found for {r_id}.")
            continue
            
        # Ensure UTC and aligned. sced_fetcher returns UTC.
        df_actual_full = df_actual_full.set_index('Time')
        
        # B. Model 1: Baseline (Fixed Tilt, No Tracking)
        print(f"  Running Baseline Model (Fixed)...")
        # Generate for both years to cover range
        p24_fixed = fetch_tmy.get_profile_for_year(2024, "Solar", capacity, lat=lat, lon=lon, tracking=False)
        p25_fixed = fetch_tmy.get_profile_for_year(2025, "Solar", capacity, lat=lat, lon=lon, tracking=False)
        prof_fixed = pd.concat([p24_fixed, p25_fixed])
        # Force unique in case of overlap (though distinct years shouldn't overlap)
        prof_fixed = prof_fixed[~prof_fixed.index.duplicated(keep='first')]

        # C. Model 2: Advanced (Single-Axis Tracking, DC/AC clipping)
        print(f"  Running Advanced Model (Tracking)...")
        p24_track = fetch_tmy.get_profile_for_year(2024, "Solar", capacity, lat=lat, lon=lon, tracking=True)
        p25_track = fetch_tmy.get_profile_for_year(2025, "Solar", capacity, lat=lat, lon=lon, tracking=True)
        prof_tracking = pd.concat([p24_track, p25_track])
        prof_tracking = prof_tracking[~prof_tracking.index.duplicated(keep='first')]
        
        # D. Align index
        fixed_aligned = prof_fixed.reindex(df_actual_full.index).fillna(0)
        tracking_aligned = prof_tracking.reindex(df_actual_full.index).fillna(0)
        
        # --- APPLY ECONOMIC DISPATCH (CURTAILMENT) ---
        # Cap modeled generation at Base Point to reflect grid limits
        if 'Base_Point_MW' in df_actual_full.columns:
            base_point = df_actual_full['Base_Point_MW'].fillna(np.inf).clip(lower=0)
            fixed_aligned = np.minimum(fixed_aligned, base_point)
            tracking_aligned = np.minimum(tracking_aligned, base_point)
        else:
            print("  ! Warning: No Base Point data found. Using unconstrained physics model.")
        
        # E. Calculate Metrics
        metrics_fixed = calculate_metrics(df_actual_full['Actual_MW'], fixed_aligned)
        metrics_tracking = calculate_metrics(df_actual_full['Actual_MW'], tracking_aligned)
        
        results.append({
            'Project': p_name,
            'Model': 'Baseline (Fixed+Curtailed)',
            'R': metrics_fixed['R'],
            'MBE (MW)': metrics_fixed['MBE'],
            'RMSE (MW)': metrics_fixed['RMSE']
        })
        results.append({
            'Project': p_name,
            'Model': 'Advanced (Tracking+Curtailed)',
            'R': metrics_tracking['R'],
            'MBE (MW)': metrics_tracking['MBE'],
            'RMSE (MW)': metrics_tracking['RMSE']
        })
        
    # Print Summary
    if results:
        df_res = pd.DataFrame(results)
        print("\n" + "="*80)
        print(f"SOLAR BENCHMARK RESULTS ({start_date} to {end_date})")
        print("="*80)
        print(df_res.to_string(index=False))
        print("\n" + "="*80)
        
        # Save to JSON for app.py
        output_path = 'benchmark_results_solar.json'
        df_res.to_json(output_path, orient='records', indent=4)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_benchmark()
