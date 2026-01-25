import pandas as pd
import numpy as np
import json
import sced_fetcher
import fetch_tmy
import os

def calculate_r_at_scales(project_name):
    """Calculates Pearson correlation at different time aggregations."""
    with open('ercot_assets.json', 'r') as f:
        assets = json.load(f)
    
    meta = assets.get(project_name)
    if not meta:
        return None
    
    # Setup - Using the same range as the main benchmark
    start_date = "2024-10-01"
    end_date = "2024-11-20"
    r_id = meta['resource_name']
    cap = meta['capacity_mw']
    lat, lon = meta['lat'], meta['lon']
    tech = meta['tech']
    
    # 1. Fetch Actual (15-min alignment)
    df_act = sced_fetcher.get_asset_period_data(r_id, start_date, end_date)
    if df_act.empty:
        return None
    df_act = df_act.set_index('Time')['Actual_MW']
    
    # 2. Generate Modeled (15-min alignment)
    if tech == 'Wind':
        hub_h = meta.get('hub_height_m', 80)
        prof = fetch_tmy.get_profile_for_year(2024, "Wind", cap, lat, lon, hub_height=hub_h)
    else:
        prof = fetch_tmy.get_profile_for_year(2024, "Solar", cap, lat, lon, tracking=True)
    
    df_mod = prof.reindex(df_act.index).fillna(0)
    
    # 3. Combine into a single dataframe
    df = pd.DataFrame({'actual': df_act, 'modeled': df_mod})
    
    # 4. Aggregations
    scales = {
        '15-Minute': df,
        'Hourly': df.resample('1H').mean(),
        'Daily': df.resample('1D').mean(),
        'Weekly': df.resample('1W').mean()
    }
    
    results = {}
    for name, scale_df in scales.items():
        # Drop any NaNs that might have been introduced during resampling
        scale_df = scale_df.dropna()
        if len(scale_df) > 1:
            results[name] = scale_df['actual'].corr(scale_df['modeled'])
        else:
            results[name] = np.nan
            
    return results

if __name__ == "__main__":
    with open('ercot_assets.json', 'r') as f:
        assets = json.load(f)
    
    projects = sorted(list(assets.keys()))
    print(f"Analyzing all {len(projects)} projects...")
    
    summary = []
    for p in projects:
        print(f"Processing {p}...")
        try:
            res = calculate_r_at_scales(p)
            if res:
                res['Project'] = p
                summary.append(res)
        except Exception as e:
            print(f"Error processing {p}: {e}")
            
    df_summary = pd.DataFrame(summary).set_index('Project')
    df_summary.to_csv('all_scales_correlation.csv')
    
    print("\n" + "="*80)
    print("CORRELATION (R) AT DIFFERENT AGGREGATION SCALES (ALL ASSETS)")
    print("="*80)
    print(df_summary.to_string())
    print("="*80)
