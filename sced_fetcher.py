import gridstatus
import pandas as pd
import os
from datetime import datetime, timedelta

CACHE_DIR = "sced_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_asset_actual_gen(resource_name, date):
    """
    Fetches actual 15-min generation for a specific resource from ERCOT 60-day disclosure.
    """
    if isinstance(date, str):
        date = pd.Timestamp(date).date()
    elif isinstance(date, datetime):
        date = date.date()
        
    cache_file = os.path.join(CACHE_DIR, f"{date}_{resource_name}.parquet")
    
    if os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Error reading cache: {e}")

    iso = gridstatus.Ercot()
    try:
        print(f"Fetching SCED disclosure for {date}...")
        data = iso.get_60_day_sced_disclosure(date=date)
        if 'sced_gen_resource' not in data:
            print(f"No generation data found in disclosure for {date}")
            return pd.DataFrame()
            
        df_gen = data['sced_gen_resource']
        # Strip column names
        df_gen.columns = [c.strip() for c in df_gen.columns]
        
        # Filter for our specific resource
        df_asset = df_gen[df_gen['Resource Name'] == resource_name].copy()
        
        if df_asset.empty:
            print(f"Resource {resource_name} not found in {date} disclosure.")
            return pd.DataFrame()
            
        # Clean and simplify
        # We want 'Interval Start' and 'Telemetered Net Output'
        # SCED data is usually every 5 mins or so, we need to resample to 15-min
        df_asset['Time'] = pd.to_datetime(df_asset['Interval Start'], utc=True)
        df_asset = df_asset.sort_values('Time')
        
        # Resample to 15-min to match model (averaging generation)
        df_resampled = df_asset.set_index('Time')['Telemetered Net Output'].resample('15min').mean().reset_index()
        df_resampled = df_resampled.rename(columns={'Telemetered Net Output': 'Actual_MW'})
        
        # Save to cache
        df_resampled.to_parquet(cache_file)
        return df_resampled
        
    except Exception as e:
        print(f"Error fetching SCED data: {e}")
        return pd.DataFrame()

def get_asset_period_data(resource_name, start_date, end_date):
    """
    Fetches actual generation for a date range.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    all_dfs = []
    
    for d in dates:
        df = get_asset_actual_gen(resource_name, d.date())
        if not df.empty:
            all_dfs.append(df)
            
    if not all_dfs:
        return pd.DataFrame()
        
    return pd.concat(all_dfs).drop_duplicates('Time').sort_values('Time')

if __name__ == "__main__":
    # Test for Frye Solar (Swisher County) - approx 65 days ago
    test_date = datetime.now() - timedelta(days=65)
    print(f"Testing fetch for Frye Solar on {test_date.date()}...")
    df = get_asset_actual_gen("FRYE_SLR_UNIT1", test_date)
    if not df.empty:
        print(df.head())
        print(f"Total points: {len(df)}")
