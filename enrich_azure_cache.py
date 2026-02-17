
import pandas as pd
import sced_fetcher
from datetime import date, datetime, timedelta
import os

CACHE_FILE = "sced_cache/AZURE_SKY_WIND_AGG_2025_full.parquet"
RESOURCE = "AZURE_SKY_WIND_AGG"
YEAR = 2025

def enrich_cache():
    print(f"Checking {CACHE_FILE}...")
    
    if not os.path.exists(CACHE_FILE):
        print("Cache file not found. Running full fetch...")
        sced_fetcher.get_asset_period_data(RESOURCE, date(YEAR, 1, 1), date.today(), require_base_point=True)
        sced_fetcher.consolidate_year(RESOURCE, YEAR)
        return

    df = pd.read_parquet(CACHE_FILE)
    print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")
    
    if 'Base_Point_MW' in df.columns:
        # Check if mostly populated
        m = df['Base_Point_MW'].notna().sum()
        print(f"Base_Point_MW present. {m}/{len(df)} non-null.")
        if m > len(df) * 0.9:
            print("Seems sufficient. Exiting.")
            return

    print("Re-fetching daily data to populate Base_Point_MW...")
    # We iterate daily to force granular fetch which captures Base Points
    # sced_fetcher.get_asset_actual_gen handles the heavy lifting
    
    start_date = date(YEAR, 1, 1)
    if YEAR <= datetime.now().year:
        end_date = date.today()
    else:
        end_date = date(YEAR, 12, 31)
        
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    dfs = []
    for d in dates:
        # This will hit daily cache if exists. 
        # If daily cache is missing Base_Point, it might re-use it?
        # sced_fetcher logic aggregates from 15-min segments.
        # We need to make sure we force a re-parse if the daily cache is "bad"
        d_str = d.strftime('%Y-%m-%d')
        # Actually, let's just use the fetcher. It should work if we ask it to.
        # But to be safe, we might need to blow away bad daily cache? 
        # Let's hope daily cache has Base Point (it usually does if `get_daily_disclosure` was used).
        
        try:
           df_d = sced_fetcher.get_asset_actual_gen(RESOURCE, d.date()) 
           if not df_d.empty:
               dfs.append(df_d)
        except Exception as e:
            print(f"Error on {d_str}: {e}")
            
    if dfs:
        print("Concatenating...")
        full_df = pd.concat(dfs).drop_duplicates('Time').sort_values('Time')
        # Normalize
        full_df = sced_fetcher._normalize_asset_cache_df(full_df)
        
        print(f"Saving enriched file with {len(full_df)} rows...")
        full_df.to_parquet(CACHE_FILE)
        print("Done.")
    else:
        print("No data fetched.")

if __name__ == "__main__":
    enrich_cache()
