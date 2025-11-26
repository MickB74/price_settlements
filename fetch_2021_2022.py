import gridstatus
import pandas as pd
import patch_gridstatus

def fetch_and_cache_years(years):
    iso = gridstatus.Ercot()
    
    for year in years:
        cache_file = f"ercot_rtm_{year}.parquet"
        
        print(f"--- Processing {year} ---")
        if pd.io.common.file_exists(cache_file):
            print(f"{year} data already cached at {cache_file}.")
            continue

        print(f"Fetching {year} data from ERCOT...")
        try:
            df = iso.get_rtm_spp(year=year)
            
            # Pre-process
            if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                df['Time'] = pd.to_datetime(df['Time'], utc=True)
            
            df['Time_Central'] = df['Time'].dt.tz_convert('US/Central')
            
            print(f"Successfully fetched {len(df)} rows for {year}.")
            
            # Save to cache
            df.to_parquet(cache_file)
            print(f"Saved to {cache_file}")
            
        except Exception as e:
            print(f"Error fetching {year} data: {e}")

if __name__ == "__main__":
    fetch_and_cache_years([2021, 2022])
