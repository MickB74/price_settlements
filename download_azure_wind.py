import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Ensure we can import sced_fetcher (in current dir)
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
import sced_fetcher

# Parameters
RESOURCES = ["VORTEX_WIND1", "VORTEX_WIND2", "VORTEX_WIND3", "VORTEX_WIND4"]
AGG_NAME = "AZURE_SKY_WIND_AGG"
YEARS = [2024, 2025]
WORKERS = 30

def prefetch_daily_disclosure(date):
    """Ensures the full daily disclosure is cached."""
    try:
        df = sced_fetcher.get_daily_disclosure(date)
        if not df.empty:
            return date, "Success"
        return date, "Empty"
    except Exception as e:
        return date, f"Error: {e}"

def extract_resource_day(args):
    """Extracts specific resource data from cached disclosure."""
    resource, date = args
    try:
        # This will use the cache we just pre-filled
        df = sced_fetcher.get_asset_actual_gen(resource, date)
        if not df.empty:
            return resource, date, df
        return resource, date, pd.DataFrame()
    except Exception as e:
        print(f"Error extracting {resource} on {date}: {e}")
        return resource, date, pd.DataFrame()

def main():
    print(f"Starting optimized download for {AGG_NAME}...")
    
    # 1. Generate list of dates
    all_dates = []
    latest_disclosure = (datetime.now() - timedelta(days=60)).date()
    START_DATE_LIMIT = datetime(2024, 1, 1).date()
    
    for year in YEARS:
        start_date = datetime(year, 1, 1).date()
        end_date = datetime(year, 12, 31).date()
        
        # Enforce start limit
        if start_date < START_DATE_LIMIT:
            start_date = START_DATE_LIMIT
        
        # Cap at 60-day lag
        if end_date > latest_disclosure:
            end_date = latest_disclosure
            
        if start_date > end_date:
            continue
            
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        all_dates.extend([d.date() for d in dates])
    
    all_dates = sorted(list(set(all_dates)))
    print(f"Total days to process: {len(all_dates)}")
    
    # 2. Pre-fetch Daily Disclosures (Network Bound)
    print("\n[Phase 1] Pre-fetching daily disclosures (Parallel)...")
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_date = {executor.submit(prefetch_daily_disclosure, d): d for d in all_dates}
        
        for i, future in enumerate(as_completed(future_to_date)):
            d = future_to_date[future]
            try:
                date, status = future.result()
                if i % 10 == 0:
                    print(f"Fetched {date}: {status} ({i+1}/{len(all_dates)})")
            except Exception as e:
                print(f"Failed {d}: {e}")

    # 3. Extract Resource Data (CPU/IO Bound)
    print("\n[Phase 2] Extracting resource data...")
    # List of (resource, date) tuples
    extraction_tasks = [(res, d) for d in all_dates for res in RESOURCES]
    
    # Dictionary to hold results: {date: {res: df}}
    data_by_date = {d: {} for d in all_dates}
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_task = {executor.submit(extract_resource_day, t): t for t in extraction_tasks}
        
        for i, future in enumerate(as_completed(future_to_task)):
            res, d, df = future.result()
            if not df.empty:
                data_by_date[d][res] = df
            
            if i % 50 == 0:
                print(f"Extracted {res} on {d} ({i+1}/{len(extraction_tasks)})")

    # 4. Aggregate and Save
    print("\n[Phase 3] Aggregating and Saving...")
    
    for year in YEARS:
        year_rows = []
        start_of_year = datetime(year, 1, 1).date()
        end_of_year = datetime(year, 12, 31).date()
        
        dates_in_year = [d for d in all_dates if start_of_year <= d <= end_of_year]
        
        if not dates_in_year:
            continue
            
        print(f"Processing Year {year} ({len(dates_in_year)} days)...")
        
        for d in dates_in_year:
            day_data = data_by_date[d]
            if not day_data:
                continue
            
            # We need to align all available resources for this day
            # Assuming they have matching 15-min intervals
            combined_day = pd.DataFrame()
            
            for res, df in day_data.items():
                if combined_day.empty:
                    combined_day = df.set_index('Time')[['Actual_MW']].rename(columns={'Actual_MW': res})
                else:
                    combined_day = combined_day.join(df.set_index('Time')[['Actual_MW']].rename(columns={'Actual_MW': res}), how='outer')
            
            if combined_day.empty:
                continue
                
            # Sum rows (treating NaN as 0)
            combined_day['Actual_MW'] = combined_day.sum(axis=1)
            
            # Reset index to get Time back
            result = combined_day[['Actual_MW']].reset_index()
            year_rows.append(result)
            
        if year_rows:
            full_year_df = pd.concat(year_rows).sort_values('Time')
            
            # Save
            cache_dir = "sced_cache"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            out_path = os.path.join(cache_dir, f"{AGG_NAME}_{year}_full.parquet")
            full_year_df.to_parquet(out_path)
            print(f"Saved {year} data to {out_path} ({len(full_year_df)} rows)")
        else:
            print(f"No data for {year}")

    print("\nDone!")

if __name__ == "__main__":
    main()
