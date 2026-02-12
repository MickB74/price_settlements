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
        # Each unit_df has columns: Time (index), Actual_MW, MWh_interval, coverage
        df_combined = pd.concat(unit_dfs, axis=1)
        
        # Summing across identical column names in a concatenated dataframe results in a new df or series
        # We handle this manually for clarity
        actual_mw_cols = [i for i, col in enumerate(df_combined.columns) if col == 'Actual_MW']
        energy_cols = [i for i, col in enumerate(df_combined.columns) if col == 'MWh_interval']
        coverage_cols = [i for i, col in enumerate(df_combined.columns) if col == 'coverage']

        df_result = pd.DataFrame({
            'Time': df_combined.index,
            'Actual_MW': df_combined.iloc[:, actual_mw_cols].sum(axis=1),
            'MWh_interval': df_combined.iloc[:, energy_cols].sum(axis=1),
            'coverage': df_combined.iloc[:, coverage_cols].mean(axis=1) # Average coverage across units
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
        
    # --- Rigorous Time-Weighted Average (TWA) Logic ---
    # 1. Clean and Prepare
    df_asset['Time'] = pd.to_datetime(df_asset['Interval Start'], utc=True)
    df_asset = df_asset.sort_values('Time').drop_duplicates('Time')
    
    # 2. Create boundary timestamps to ensure segments don't cross 15-min boundaries
    start_bound = df_asset['Time'].min().floor('15min')
    end_bound = df_asset['Time'].max().ceil('15min')
    boundaries = pd.date_range(start=start_bound, end=end_bound, freq='15min', tz='UTC')
    
    # 3. Merge actual timestamps with boundary timestamps
    all_ts = sorted(list(set(df_asset['Time'].tolist() + boundaries.tolist())))
    
    # 4. Reindex using forward-fill (stepwise constant assumption)
    # This ensures we have a MW value at every 15-minute boundary
    df_step = df_asset.set_index('Time')[['Telemetered Net Output']].reindex(all_ts).ffill()
    df_step.index.name = 'Time'
    df_step = df_step.reset_index()
    
    # 5. Calculate durations and energy per segment
    df_step['next_time'] = df_step['Time'].shift(-1)
    df_step['duration_sec'] = (df_step['next_time'] - df_step['Time']).dt.total_seconds()
    
    # Guardrail: Drop last record, ignore non-positive durations, cap large gaps
    df_step = df_step.dropna(subset=['next_time', 'duration_sec'])
    df_step = df_step[df_step['duration_sec'] > 0].copy()
    
    # Cap gaps at 1 hour (3600s) to prevent runaway energy integration during outages
    df_step['duration_sec'] = df_step['duration_sec'].clip(upper=3600)
    
    # Calculate Segment Energy (MWh)
    df_step['energy_mwh'] = df_step['Telemetered Net Output'] * (df_step['duration_sec'] / 3600.0)
    
    # 6. Aggregate to 15-minute Intervals
    # Assign each segment to its 15-min bucket start
    df_step['Interval_Start'] = df_step['Time'].dt.floor('15min')
    
    agg_funcs = {
        'energy_mwh': 'sum',
        'duration_sec': 'sum'
    }
    
    # If we want to keep Base Point, handle it too (simple mean usually okay for BP BP is a target)
    if 'Base Point' in df_asset.columns:
        # We'd need to re-step Base Point too if we wanted it perfect. 
        # For now, let's just focus on Actuals.
        pass

    results = df_step.groupby('Interval_Start').agg(agg_funcs).reset_index()
    
    # 7. Final Metrics
    results['coverage'] = results['duration_sec'] / 900.0
    
    # Actual_MW (TWA) = Total Energy / (Total Time in hours)
    results['Actual_MW'] = results['energy_mwh'] / (results['duration_sec'] / 3600.0)
    results = results.rename(columns={'Interval_Start': 'Time', 'energy_mwh': 'MWh_interval'})
    
    # Ensure columns match expectations
    df_resampled = results[['Time', 'Actual_MW', 'MWh_interval', 'coverage']].copy()
    
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
