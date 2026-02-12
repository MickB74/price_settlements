import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure we can import sced_fetcher (in current dir)
import sced_fetcher

# Parameters
RESOURCES = ["VORTEX_WIND1", "VORTEX_WIND2", "VORTEX_WIND3", "VORTEX_WIND4"]
AGG_NAME = "AZURE_SKY_WIND_AGG"
WORKERS = 20

# Targeted Dates based on gap analysis
MISSING_RANGES = [
    ("2025-10-07", "2025-10-07"),
    ("2025-12-04", "2025-12-31")
]

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
        df = sced_fetcher.get_asset_actual_gen(resource, date)
        if not df.empty:
            return resource, date, df
        return resource, date, pd.DataFrame()
    except Exception as e:
        print(f"Error extracting {resource} on {date}: {e}")
        return resource, date, pd.DataFrame()

def main():
    print(f"Filling missing data for {AGG_NAME}...")
    
    # 1. Generate list of dates
    all_dates = []
    for start_str, end_str in MISSING_RANGES:
        s = datetime.strptime(start_str, "%Y-%m-%d").date()
        e = datetime.strptime(end_str, "%Y-%m-%d").date()
        dates = pd.date_range(start=s, end=e, freq='D')
        all_dates.extend([d.date() for d in dates])
    
    all_dates = sorted(list(set(all_dates)))
    print(f"Total missing days to process: {len(all_dates)}")
    
    # 2. Pre-fetch Daily Disclosures
    print("\n[Phase 1] Pre-fetching daily disclosures...")
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_date = {executor.submit(prefetch_daily_disclosure, d): d for d in all_dates}
        
        for i, future in enumerate(as_completed(future_to_date)):
            d = future_to_date[future]
            try:
                date, status = future.result()
                print(f"Fetched {date}: {status}")
            except Exception as e:
                print(f"Failed {d}: {e}")

    # 3. Extract Resource Data
    print("\n[Phase 2] Extracting resource data...")
    extraction_tasks = [(res, d) for d in all_dates for res in RESOURCES]
    data_by_date = {d: {} for d in all_dates}
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_task = {executor.submit(extract_resource_day, t): t for t in extraction_tasks}
        for future in as_completed(future_to_task):
            res, d, df = future.result()
            if not df.empty:
                data_by_date[d][res] = df

    # 4. Aggregate and Save (Append to existing logic would be complex, 
    # instead we will just re-run the full aggregator script separately 
    # since it scans the cache folder).
    print("\nData collected. You should run aggregate_azure.py to rebuild the parquet.")

if __name__ == "__main__":
    main()
