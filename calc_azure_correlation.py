import pandas as pd
import numpy as np
import os
import sys

# Add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
CACHE_DIR = "sced_cache"
AZURE_FILE_2024 = os.path.join(CACHE_DIR, "AZURE_SKY_WIND_AGG_2024_full.parquet")
AZURE_FILE_2025 = os.path.join(CACHE_DIR, "AZURE_SKY_WIND_AGG_2025_full.parquet")

# We need the synthetic profile to compare against.
# In app.py, this comes from `df_primary['Gen_MW']` which is derived from `utils.generate_dummy_generation_profile`
# or a static profile. 
# For now, let's just inspect the auto-correlation or consistency of the downloaded data itself 
# OR try to load the mock profile if available.

# Actually, the user likely saw a "Correlation" metric in the app for other projects and wants to know Azure's.
# The app benchmarks against a "High Fidelity Synthetic Model".
# Let's see if we can instantiate that model or if there's a reference file.



def main():
    print("Calculating Azure Sky Wind Correlation...")
    
    # Load Actuals
    dfs = []
    if os.path.exists(AZURE_FILE_2024):
        dfs.append(pd.read_parquet(AZURE_FILE_2024))
    if os.path.exists(AZURE_FILE_2025):
        dfs.append(pd.read_parquet(AZURE_FILE_2025))
        
    if not dfs:
        print("No Azure data found.")
        return
        
    df_actual = pd.concat(dfs).sort_values("Time").set_index("Time")
    df_actual = df_actual.rename(columns={"Actual_MW": "Actual"})
    
    # Constrain to 2024 for a clean year benchmark (or overlap)
    # The synthetic model usually covers a specific year.
    # Let's assume 2024 benchmark.
    
    df_2024 = df_actual[df_actual.index.year == 2024].copy()
    
    if df_2024.empty:
        print("No 2024 data for benchmark.")
        return

    # Generate Synthetic (Mock)
    # Capacity = 350 MW
    from fetch_tmy import get_blended_profile_for_year
    
    # Azure Sky Details
    LAT = 33.1534
    LON = -99.2847
    
    # Mixed Fleet Configuration (Source: Public Project Data)
    TURBINES = [
        {'type': 'NORDEX_N149', 'count': 65, 'capacity_mw': 4.5, 'hub_height_m': 105.0},
        {'type': 'VESTAS_V163', 'count': 7,  'capacity_mw': 3.45, 'hub_height_m': 82.0}, # V136 proxy (use V163 curve)
        {'type': 'GENERIC',     'count': 7,  'capacity_mw': 2.0,  'hub_height_m': 80.0}, # V110 proxy (IEC2)
    ]
    CAPACITY = sum(t['count'] * t['capacity_mw'] for t in TURBINES) # ~330.65 MW
    
    # Loop through years
    for target_year in [2024, 2025]:
        print(f"\n--- Processing {target_year} ---")
        
        df_target = df_actual[df_actual.index.year == target_year].copy()
        
        if df_target.empty:
            print(f"No actual data for {target_year}.")
            continue
            
        # Deduplicate index if needed
        if df_target.index.has_duplicates:
            print(f"Warning: Found {df_target.index.duplicated().sum()} duplicate timestamps in {target_year} data. Keeping first.")
            df_target = df_target[~df_target.index.duplicated(keep='first')]

        print(f"Generating blended synthetic profile for {target_year}...")
        
        s_synthetic = get_blended_profile_for_year(
            year=target_year,
            tech="Wind",
            turbines=TURBINES,
            lat=LAT,
            lon=LON,
            efficiency=0.85
        )
        
        if s_synthetic.empty:
            print(f"Failed to generate synthetic profile for {target_year}.")
            continue
            
        # Align Data
        if df_target.index.tz is None:
            df_target.index = df_target.index.tz_localize("US/Central")
        else:
            df_target.index = df_target.index.tz_convert("US/Central")
            
        s_synthetic = s_synthetic.tz_convert("US/Central")
        
        # Merge
        combined = pd.DataFrame({"Actual": df_target["Actual"], "Synthetic": s_synthetic})
        combined = combined.dropna()
        
        if combined.empty:
            print(f"No overlapping data found for {target_year}.")
            continue
            
        # Calculate Correlation
        r_15min = combined["Actual"].corr(combined["Synthetic"])
        
        # Hourly
        combined_hourly = combined.resample("1h").mean()
        r_hourly = combined_hourly["Actual"].corr(combined_hourly["Synthetic"])
        
        # Daily
        combined_daily = combined.resample("1D").sum()
        r_daily = combined_daily["Actual"].corr(combined_daily["Synthetic"])
        
        print(f"--- Azure Sky Wind Correlation ({target_year}) ---")
        print(f"Data Points: {len(combined)}")
        print(f"15-min R: {r_15min:.3f}")
        print(f"Hourly R: {r_hourly:.3f}")
        print(f"Daily  R: {r_daily:.3f}")
        
        # Also stats on Capacity Factor
        cf_actual = combined["Actual"].mean() / CAPACITY
        cf_synth = combined["Synthetic"].mean() / CAPACITY
        print(f"Capacity Factor:")
        print(f"  Actual:    {cf_actual:.1%}")
        print(f"  Synthetic: {cf_synth:.1%}")

if __name__ == "__main__":
    main()
