import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import fetch_tmy
import sced_fetcher
from utils import power_curves
from utils.wind_calibration import get_offline_threshold_mw

def calculate_metrics(actual, modeled, capacity_mw=None):
    """Calculate R, MBE, and RMSE between actual and modeled series."""
    # Ensure they are the same length and aligned
    combined = pd.DataFrame({'actual': actual, 'modeled': modeled}).dropna()
    if combined.empty:
        return {'R': np.nan, 'R_Hourly': np.nan, 'R_Daily': np.nan, 'MBE': np.nan, 'RMSE': np.nan}
        
    actual = combined['actual']
    modeled = combined['modeled']
    
    # --- FILTER: Exclude "Offline" periods ---
    # User concern: "Offline actual project skewing numbers"
    # If Actual is ~0 but Model predicts significant generation, assume outage/maintenance.
    # Exclude these from R/Bias calc as they reflect operational status, not model accuracy.
    
    threshold_mw = get_offline_threshold_mw(capacity_mw)
    # Mask: Keep only points where (Actual > 0) OR (Model is also low)
    # i.e. Throw away points where (Actual == 0 AND Model > capacity-aware threshold)
    
    valid_mask = ~((actual < 0.5) & (modeled > threshold_mw))
    
    actual_filtered = actual[valid_mask]
    modeled_filtered = modeled[valid_mask]
    
    if len(actual_filtered) < 10:
        return {'R': np.nan, 'R_Hourly': np.nan, 'R_Daily': np.nan, 'MBE': np.nan, 'RMSE': np.nan}
    
    # 1. 15-Minute Correlation
    r_15min = actual_filtered.corr(modeled_filtered) if actual_filtered.nunique() > 1 and modeled_filtered.nunique() > 1 else np.nan
    
    # 2. Hourly Correlation (Resample)
    # Need datetime index for resampling. 'actual' is likely Series with numeric index if dropped na?
    # actually combined has index from input.
    combined_filtered = combined[valid_mask]
    
    # Resample to Hourly
    hourly = combined_filtered.resample('h').mean().dropna()
    r_hourly = hourly['actual'].corr(hourly['modeled']) if len(hourly) > 2 and hourly['actual'].nunique() > 1 and hourly['modeled'].nunique() > 1 else np.nan
    
    # 3. Daily Correlation (Resample)
    daily = combined_filtered.resample('D').mean().dropna()
    r_daily = daily['actual'].corr(daily['modeled']) if len(daily) > 2 and daily['actual'].nunique() > 1 and daily['modeled'].nunique() > 1 else np.nan
    
    # Mean Bias Error (MBE) - average difference
    mbe = (modeled_filtered - actual_filtered).mean()
    
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(((modeled_filtered - actual_filtered)**2).mean())
    
    return {
        'R': r_15min,
        'R_Hourly': r_hourly,
        'R_Daily': r_daily,
        'MBE': mbe,
        'RMSE': rmse,
        'Capacity Factor (Actual)': actual_filtered.mean() / actual_filtered.max() if actual_filtered.max() > 0 else 0,
        'Capacity Factor (Modeled)': modeled_filtered.mean() / modeled_filtered.max() if modeled_filtered.max() > 0 else 0
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
    
    # Range for benchmark (12 Months: Dec 2024 - Nov 2025)
    start_date = "2024-12-01"
    end_date = "2025-11-30"
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
        df_actual_full = sced_fetcher.get_asset_period_data(r_id, start_date, end_date)
        if df_actual_full.empty:
            print(f"  ! No actual data found for {r_id} in period.")
            continue
            
        # Ensure UTC and aligned. sced_fetcher returns UTC.
        df_actual_full = df_actual_full.set_index('Time')
        
        # B. Model 1: Baseline (80m, Generic Curve)
        print(f"  Running Baseline Model (80m, Generic)...")
        p24_base = fetch_tmy.get_profile_for_year(
            2024, "Wind", capacity, lat=lat, lon=lon, hub_height=80, turbine_type="GENERIC", apply_wind_calibration=False
        )
        p25_base = fetch_tmy.get_profile_for_year(
            2025, "Wind", capacity, lat=lat, lon=lon, hub_height=80, turbine_type="GENERIC", apply_wind_calibration=False
        )
        prof_baseline = pd.concat([p24_base, p25_base])
        prof_baseline = prof_baseline[~prof_baseline.index.duplicated(keep='first')]

        # C. Model 2: Advanced (Actual Hub, Actual Curve / Mixed Fleet)
        print(f"  Running Advanced Model ({actual_hub_h}m, {actual_tech_type})...")
        
        turbines = meta.get('turbines')
        if turbines:
            print(f"  ! Mixed fleet detected for {p_name}. Generating blended profile...")
            p24_adv = fetch_tmy.get_blended_profile_for_year(
                2024, "Wind", turbines, lat=lat, lon=lon
            )
            p25_adv = fetch_tmy.get_blended_profile_for_year(
                2025, "Wind", turbines, lat=lat, lon=lon
            )
        else:
            p24_adv = fetch_tmy.get_profile_for_year(
                2024, "Wind", capacity, lat=lat, lon=lon, hub_height=actual_hub_h,
                turbine_type=actual_tech_type, apply_wind_calibration=False,
            )
            p25_adv = fetch_tmy.get_profile_for_year(
                2025, "Wind", capacity, lat=lat, lon=lon, hub_height=actual_hub_h,
                turbine_type=actual_tech_type, apply_wind_calibration=False,
            )
            
        prof_advanced = pd.concat([p24_adv, p25_adv])
        prof_advanced = prof_advanced[~prof_advanced.index.duplicated(keep='first')]
        
        # D. Align index for comparison
        # Model profiles have a specific year index, actual might start at a specific day.
        # Ensure both are aligned to UTC.
        baseline_aligned = prof_baseline.reindex(df_actual_full.index).fillna(0)
        advanced_aligned = prof_advanced.reindex(df_actual_full.index).fillna(0)
        
        # --- APPLY ECONOMIC DISPATCH (CURTAILMENT) ---
        # Cap modeled generation at Base Point if available
        if 'Base_Point_MW' in df_actual_full.columns:
            # Fill NaN Base Points with infinity (unconstrained) to be safe
            base_point = df_actual_full['Base_Point_MW'].fillna(np.inf)
            base_point = base_point.clip(lower=0) # ensure no negatives
            
            baseline_aligned = np.minimum(baseline_aligned, base_point)
            advanced_aligned = np.minimum(advanced_aligned, base_point)
        else:
            print("  ! Warning: No Base Point data found. Using unconstrained physics model.")

        # E. Calculate Metrics
        metrics_baseline = calculate_metrics(df_actual_full['Actual_MW'], baseline_aligned, capacity_mw=capacity)
        metrics_advanced = calculate_metrics(df_actual_full['Actual_MW'], advanced_aligned, capacity_mw=capacity)
        
        results.append({
            'Project': p_name,
            'Model': 'Baseline (Curtailed)',
            'R': metrics_baseline['R'],
            'R_Hourly': metrics_baseline['R_Hourly'],
            'R_Daily': metrics_baseline['R_Daily'],
            'MBE (MW)': metrics_baseline['MBE'],
            'RMSE (MW)': metrics_baseline['RMSE']
        })
        results.append({
            'Project': p_name,
            'Model': f"Advanced (Curtailed)",
            'R': metrics_advanced['R'],
            'R_Hourly': metrics_advanced['R_Hourly'],
            'R_Daily': metrics_advanced['R_Daily'],
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
