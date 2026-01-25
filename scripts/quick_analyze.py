import sys
import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch_tmy
import sced_fetcher

def quick_analyze(resource_name, year, tech, capacity_mw, lat, lon, turbine_type="GENERIC"):
    print(f"--- Quick Analysis for {resource_name} ({year}) ---")
    print(f"Turbine Type: {turbine_type}")
    
    # 1. Load from Cache Files Directly or Consolidated
    dfs = []
    
    # Check consolidated first
    cons_file = os.path.join(sced_fetcher.CACHE_DIR, f"{resource_name}_{year}_full.parquet")
    if os.path.exists(cons_file):
        print("Loading from consolidated cache...")
        dfs.append(pd.read_parquet(cons_file))
    else:
        # Fallback to daily
        cache_pattern = os.path.join(sced_fetcher.CACHE_DIR, f"*{year}-*_{resource_name}.parquet")
        files = glob.glob(cache_pattern)
        print(f"Found {len(files)} cached days.")
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except: pass

    if not dfs:
        print("No cached data found.")
        return

    df_actual = pd.concat(dfs).drop_duplicates('Time').sort_values('Time')
    print(f"Loaded {len(df_actual)} actual data points.")
    
    # 2. Run Model
    print("Generating Model Profile (ERA5)...")
    s_modeled = fetch_tmy.get_profile_for_year(year, tech, capacity_mw, lat, lon, turbine_type=turbine_type)
    
    # 3. Align
    df_modeled = s_modeled.to_frame(name='Modeled_MW')
    df_modeled.index = pd.to_datetime(df_modeled.index, utc=True)
    
    # Merge on Time
    df_comp = pd.merge(df_actual, df_modeled, left_on='Time', right_index=True, how='inner')
    
    # 5. Metrics
    mae = (df_comp['Actual_MW'] - df_comp['Modeled_MW']).abs().mean()
    correlation = df_comp['Actual_MW'].corr(df_comp['Modeled_MW'])
    r2 = correlation ** 2
    
    print("\n=== PERFORMANCE RESULTS ===")
    print(f"Data Coverage:        {len(df_comp) / (365*24*4):.1%} of year")
    print(f"Correlation (R):      {correlation:.4f}")
    print(f"R-Squared (R²):       {r2:.4f}")
    print(f"Mean Abs Error (MAE): {mae:.2f} MW")
    
    # Monthly Correlation
    df_comp['Month'] = df_comp['Time'].dt.month
    monthly_r2 = df_comp.groupby('Month').apply(lambda x: x['Actual_MW'].corr(x['Modeled_MW'])**2)
    print("\n--- Monthly R² ---")
    print(monthly_r2)

if __name__ == "__main__":
    # Load Registry to get enriched turbine types
    import json
    assets_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ercot_assets.json")
    with open(assets_file, 'r') as f:
        registry = json.load(f)
        
    print("\n--- Running Analysis with USWTDB Enriched Metadata ---")
    
    # Analyze Ajax Wind (Should be GE 2.82)
    asset = registry.get("Ajax Wind")
    if asset:
        t_model = asset.get('turbine_model', asset.get('turbine_type', 'GENERIC'))
        p_name = asset.get('project_name', 'Ajax Wind')
        print(f"\nAnalyzing {p_name} with discovered turbine: {t_model}")
        quick_analyze(asset['resource_name'], 2024, "Wind", asset['capacity_mw'], asset['lat'], asset['lon'], turbine_type=t_model)
        
    # Analyze Monte Cristo (Should be Vestas V110)
    asset = registry.get("Monte Cristo Wind")
    if asset:
        t_model = asset.get('turbine_model', asset.get('turbine_type', 'GENERIC'))
        p_name = asset.get('project_name', 'Monte Cristo Wind')
        print(f"\nAnalyzing {p_name} with discovered turbine: {t_model}")
        quick_analyze(asset['resource_name'], 2024, "Wind", asset['capacity_mw'], asset['lat'], asset['lon'], turbine_type=t_model)
