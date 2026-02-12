import pandas as pd
import os

CACHE_FILE = "sced_cache/AZURE_SKY_WIND_AGG_2025_full.parquet"

if os.path.exists(CACHE_FILE):
    df = pd.read_parquet(CACHE_FILE)
    print(f"--- 2025 Data Inspection ---")
    print(f"Total Rows: {len(df)}")
    print(f"Start: {df['Time'].min()}")
    print(f"End:   {df['Time'].max()}")
    
    # Check for gaps (expecting 15-min intervals)
    full_idx = pd.date_range(start="2025-01-01 00:00", end="2025-12-31 23:45", freq="15min", tz="US/Central")
    
    # Convert df['Time'] to consistent TZ if needed
    # The parquet should have preserved TZ usually, or is UTC.
    # aggregate_azure.py used: df_vortex['Time'] = pd.to_datetime(..., utc=True)
    
    # Let's align to check gaps
    df = df.set_index('Time')
    if df.index.tz is not None:
        df_idx = df.index.tz_convert("US/Central")
    else:
        df_idx = df.index.tz_localize("UTC").tz_convert("US/Central")
        
    missing = full_idx.difference(df_idx)
    print(f"Missing Intervals: {len(missing)}")
    if len(missing) > 0:
        print(f"First 5 missing: {missing[:5]}")
        print(f"Last 5 missing: {missing[-5:]}")
else:
    print(f"File not found: {CACHE_FILE}")
