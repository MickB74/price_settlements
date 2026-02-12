import pandas as pd
import numpy as np
import os
import sys

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sced_fetcher
from datetime import datetime, timedelta

def compare_methods(resource_name, date_str):
    # 1. Fetch raw full disclosure
    date = pd.to_datetime(date_str).date()
    df_full = sced_fetcher.get_daily_disclosure(date)
    if df_full.empty:
        return None
        
    # Handle Azure Sky aggregation if needed
    if resource_name == "AZURE_SKY_WIND_AGG":
        # Aggregate all 4 VORTEX units
        v_units = ["VORTEX_WIND1", "VORTEX_WIND2", "VORTEX_WIND3", "VORTEX_WIND4"]
        df_asset = df_full[df_full['Resource Name'].isin(v_units)].copy()
        if df_asset.empty: return None
        df_asset['Time'] = pd.to_datetime(df_asset['Interval Start'], utc=True)
        # Sum MW by timestamp
        df_asset = df_asset.groupby('Time')['Telemetered Net Output'].sum().reset_index()
    else:
        df_asset = df_full[df_full['Resource Name'] == resource_name].copy()
        if df_asset.empty: return None
        df_asset['Time'] = pd.to_datetime(df_asset['Interval Start'], utc=True)
        df_asset = df_asset.sort_values('Time').drop_duplicates('Time')
    
    # --- METHOD 1: PRIOR (Simple Mean) ---
    # Group by 15-min interval and take mean
    df_prior = df_asset.copy()
    df_prior['Interval_Start'] = df_prior['Time'].dt.floor('15min')
    prior_agg = df_prior.groupby('Interval_Start')['Telemetered Net Output'].mean() * 0.25
    total_mwh_prior = prior_agg.sum()
    
    # --- METHOD 2: CURRENT (TWA) ---
    # Re-run the actual logic from sced_fetcher (re-implemented here for clarity/verification)
    start_bound = df_asset['Time'].min().floor('15min')
    end_bound = df_asset['Time'].max().ceil('15min')
    boundaries = pd.date_range(start=start_bound, end=end_bound, freq='15min', tz='UTC')
    all_ts = sorted(list(set(df_asset['Time'].tolist() + boundaries.tolist())))
    
    df_step = df_asset.set_index('Time')[['Telemetered Net Output']].reindex(all_ts).ffill()
    df_step = df_step.reset_index()
    df_step['next_time'] = df_step['Time'].shift(-1)
    df_step['duration_sec'] = (df_step['next_time'] - df_step['Time']).dt.total_seconds()
    df_step = df_step.dropna(subset=['next_time', 'duration_sec'])
    df_step = df_step[df_step['duration_sec'] > 0].copy()
    df_step['duration_sec'] = df_step['duration_sec'].clip(upper=3600)
    df_step['energy_mwh'] = df_step['Telemetered Net Output'] * (df_step['duration_sec'] / 3600.0)
    
    total_mwh_twa = df_step['energy_mwh'].sum()
    
    return {
        'date': date_str,
        'prior_mwh': total_mwh_prior,
        'twa_mwh': total_mwh_twa,
        'diff_mwh': total_mwh_twa - total_mwh_prior,
        'pct_diff': (total_mwh_twa / total_mwh_prior - 1) * 100 if total_mwh_prior != 0 else 0
    }

if __name__ == "__main__":
    resource = "AZURE_SKY_WIND_AGG"
    # Test a few days across 2024 to get a representative sample
    test_days = ["2024-01-15", "2024-04-20", "2024-07-10", "2024-10-05"]
    
    results = []
    for day in test_days:
        print(f"Analyzing {day}...")
        res = compare_methods(resource, day)
        if res:
            results.append(res)
            
    df_res = pd.DataFrame(results)
    print("\n--- TWA vs Prior (Simple Mean) Comparison ---")
    print(df_res)
    
    avg_diff = df_res['pct_diff'].mean()
    print(f"\nAverage Percentage Change: {avg_diff:+.2f}%")
    
    # Scale to full year estimate
    # 2024 total energy was approx 1.1M MWh (based on CFS check earlier)
    # Let's see the impact.
