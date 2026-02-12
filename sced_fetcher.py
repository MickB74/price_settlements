import gridstatus
import pandas as pd
import os
from datetime import datetime, timedelta

CACHE_DIR = "sced_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_daily_disclosure(date):
    """
    Fetches and caches the FULL daily SCED disclosure dataframe.
    """
    if isinstance(date, str):
        date = pd.Timestamp(date).date()
    elif isinstance(date, datetime):
        date = date.date()
        
    cache_file = os.path.join(CACHE_DIR, f"full_disclosure_{date}.parquet")
    
    if os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except:
            pass
            
    iso = gridstatus.Ercot()
    try:
        print(f"Fetching full SCED disclosure for {date}...")
        data = iso.get_60_day_sced_disclosure(date=date)
        if 'sced_gen_resource' not in data:
            return pd.DataFrame()
            
        df_gen = data['sced_gen_resource']
        df_gen.columns = [c.strip() for c in df_gen.columns]
        
        # Save full disclosure to cache
        df_gen.to_parquet(cache_file)
        return df_gen
    except Exception as e:
        print(f"Error fetching daily disclosure: {e}")
        return pd.DataFrame()

def get_asset_actual_gen(resource_name, date):
    """
    Fetches actual 15-min generation for a specific resource.
    Uses daily disclosure cache for performance.
    Special handling for AZURE_SKY_WIND_AGG: aggregates all 4 VORTEX units.
    """
    if isinstance(date, str):
        date = pd.Timestamp(date).date()
    elif isinstance(date, datetime):
        date = date.date()
        
    # Check if we already have the asset-specific cache
    asset_cache = os.path.join(CACHE_DIR, f"{date}_{resource_name}.parquet")
    if os.path.exists(asset_cache):
        try:
            return pd.read_parquet(asset_cache)
        except:
            pass
    
    # Special handling for Azure Sky Wind aggregation
    if resource_name == "AZURE_SKY_WIND_AGG":
        vortex_units = ["VORTEX_WIND1", "VORTEX_WIND2", "VORTEX_WIND3", "VORTEX_WIND4"]
        unit_dfs = []
        
        for unit in vortex_units:
            df_unit = get_asset_actual_gen(unit, date)
            if not df_unit.empty:
                unit_dfs.append(df_unit.set_index('Time'))
        
        if not unit_dfs:
            return pd.DataFrame()
        
        # Combine all units and sum their generation
        df_combined = pd.concat(unit_dfs, axis=1)
        
        # Sum Actual_MW columns (there will be multiple with same name)
        actual_cols = [col for col in df_combined.columns if 'Actual_MW' in str(col)]
        df_result = pd.DataFrame({
            'Time': df_combined.index,
            'Actual_MW': df_combined[actual_cols].sum(axis=1)
        }).reset_index(drop=True)
        
        # Save aggregated cache
        df_result.to_parquet(asset_cache)
        return df_result
        
    # Otherwise, get the full daily disclosure (potentially from cache)
    df_full = get_daily_disclosure(date)
    if df_full.empty:
        return pd.DataFrame()
        
    # Filter for our specific resource
    df_asset = df_full[df_full['Resource Name'] == resource_name].copy()
    
    if df_asset.empty:
        print(f"Resource {resource_name} not found in {date} disclosure.")
        # Save empty to avoid re-searching? 
        # No, might be too many files.
        return pd.DataFrame()
        
    # Clean and simplify
    df_asset['Time'] = pd.to_datetime(df_asset['Interval Start'], utc=True)
    df_asset = df_asset.sort_values('Time')
    
    cols_to_resample = ['Telemetered Net Output']
    rename_map = {'Telemetered Net Output': 'Actual_MW'}
    
    if 'Base Point' in df_asset.columns:
        cols_to_resample.append('Base Point')
        rename_map['Base Point'] = 'Base_Point_MW'
        
    df_resampled = df_asset.set_index('Time')[cols_to_resample].resample('15min').mean().reset_index()
    df_resampled = df_resampled.rename(columns=rename_map)
    
    # Save asset-specific cache
    df_resampled.to_parquet(asset_cache)
    return df_resampled

def get_asset_period_data(resource_name, start_date, end_date):
    """
    Fetches actual generation for a date range.
    Optimized to check for consolidated yearly cache first.
    """
    if isinstance(start_date, str): start_date = pd.Timestamp(start_date).date()
    if isinstance(end_date, str): end_date = pd.Timestamp(end_date).date()
    
    years = range(start_date.year, end_date.year + 1)
    all_dfs = []

    for y in years:
        year_cache = os.path.join(CACHE_DIR, f"{resource_name}_{y}_full.parquet")
        
        # 1. Try Consolidated Year Cache
        if os.path.exists(year_cache):
            try:
                df_year = pd.read_parquet(year_cache)
                # Filter to requested range within this year
                mask = (df_year['Time'].dt.date >= start_date) & (df_year['Time'].dt.date <= end_date)
                all_dfs.append(df_year.loc[mask])
                continue
            except:
                pass # Fallback if corrupt
        
        # 2. Fallback to Daily files for this year part
        y_start = max(start_date, datetime(y, 1, 1).date())
        y_end = min(end_date, datetime(y, 12, 31).date())
        
        dates = pd.date_range(start=y_start, end=y_end, freq='D')
        for d in dates:
            df = get_asset_actual_gen(resource_name, d.date())
            if not df.empty:
                all_dfs.append(df)
            
    if not all_dfs:
        return pd.DataFrame()
        
    return pd.concat(all_dfs).drop_duplicates('Time').sort_values('Time')

def consolidate_year(resource_name, year):
    """Helper to merge daily files into one fast year file."""
    start = datetime(year, 1, 1).date()
    end = datetime(year, 12, 31).date()
    
    # We use the internal daily fetcher but bypassing the consolidated check to avoid recursion logic
    # Actually we can just iterate files manually or use the slow path
    dates = pd.date_range(start=start, end=end, freq='D')
    dfs = []
    for d in dates:
        f = os.path.join(CACHE_DIR, f"{d.date()}_{resource_name}.parquet")
        if os.path.exists(f):
            try:
                dfs.append(pd.read_parquet(f))
            except: pass
            
    if dfs:
        full_df = pd.concat(dfs).drop_duplicates('Time').sort_values('Time')
        out_path = os.path.join(CACHE_DIR, f"{resource_name}_{year}_full.parquet")
        full_df.to_parquet(out_path)
        print(f"Consolidated {year} cache for {resource_name}: {len(full_df)} rows")
        return True
    return False


if __name__ == "__main__":
    # Test for Frye Solar (Swisher County) - approx 65 days ago
    test_date = datetime.now() - timedelta(days=65)
    print(f"Testing fetch for Frye Solar on {test_date.date()}...")
    df = get_asset_actual_gen("FRYE_SLR_UNIT1", test_date)
    if not df.empty:
        print(df.head())
        print(f"Total points: {len(df)}")
