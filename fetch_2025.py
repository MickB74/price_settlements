import gridstatus
import pandas as pd
import patch_gridstatus
import os
from datetime import datetime

def fetch_and_update_2025():
    year = 2025
    cache_file = f"ercot_rtm_{year}.parquet"
    
    # Remove existing cache to force update
    if os.path.exists(cache_file):
        print(f"Removing outdated cache file: {cache_file}")
        # Load old max date just for info
        try:
            old_df = pd.read_parquet(cache_file)
            print(f"Old data max date: {old_df['Time'].max()}")
        except:
            pass
        os.remove(cache_file)
    
    print(f"Fetching fresh {year} data from ERCOT...")
    iso = gridstatus.Ercot()
    
    try:
        # gridstatus handles the fetching
        df = iso.get_rtm_spp(year=year)
        
        # Pre-process (Standardize time columns)
        if not pd.api.types.is_datetime64_any_dtype(df['Time']):
            df['Time'] = pd.to_datetime(df['Time'], utc=True)
        
        df['Time_Central'] = df['Time'].dt.tz_convert('US/Central')
        
        print(f"Successfully fetched {len(df)} rows.")
        print(f"New data range: {df['Time_Central'].min()} to {df['Time_Central'].max()}")
        
        # Save to cache
        df.to_parquet(cache_file)
        print(f"Saved fresh data to {cache_file}")
        
    except Exception as e:
        print(f"Error fetching {year} data: {e}")

if __name__ == "__main__":
    fetch_and_update_2025()
