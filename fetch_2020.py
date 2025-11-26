import gridstatus
import pandas as pd
import patch_gridstatus

def fetch_and_cache_2020():
    year = 2020
    cache_file = f"ercot_rtm_{year}.parquet"
    
    print(f"Checking for {cache_file}...")
    if pd.io.common.file_exists(cache_file):
        print("2020 data already cached.")
        df = pd.read_parquet(cache_file)
        print(f"Loaded {len(df)} rows.")
        return

    print("Fetching 2020 data from ERCOT (this may take a moment)...")
    iso = gridstatus.Ercot()
    try:
        df = iso.get_rtm_spp(year=year)
        
        # Pre-process (same as app.py)
        if not pd.api.types.is_datetime64_any_dtype(df['Time']):
            df['Time'] = pd.to_datetime(df['Time'], utc=True)
        
        df['Time_Central'] = df['Time'].dt.tz_convert('US/Central')
        
        print(f"Successfully fetched {len(df)} rows.")
        print(df.head())
        
        # Save to cache
        df.to_parquet(cache_file)
        print(f"Saved to {cache_file}")
        
    except Exception as e:
        print(f"Error fetching 2020 data: {e}")

if __name__ == "__main__":
    fetch_and_cache_2020()
