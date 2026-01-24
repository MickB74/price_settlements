import sys
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import pandas as pd
import time

# Add parent directory to path so we can import sced_fetcher
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sced_fetcher

def fetch_worker(args):
    """Worker function for threading."""
    resource, date = args
    try:
        # Check cache first to avoid print noise if already exists
        cache_path = os.path.join(sced_fetcher.CACHE_DIR, f"{date}_{resource}.parquet")
        if os.path.exists(cache_path):
            return date, "Cached", 0
            
        print(f"[{date}] Starting fetch...")
        df = sced_fetcher.get_asset_actual_gen(resource, date)
        if not df.empty:
            return date, "Success", len(df)
        else:
            return date, "Empty", 0
    except Exception as e:
        return date, f"Error: {e}", 0

def main():
    parser = argparse.ArgumentParser(description="Bulk fetch SCED data for a resource.")
    parser.add_argument("resource", help="Resource ID (e.g. FRYE_SLR_UNIT1)")
    parser.add_argument("--year", type=int, default=2024, help="Year to fetch (default: 2024)")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    args = parser.parse_args()

    resource = args.resource
    year = args.year
    
    # Define range
    start_date = datetime(year, 1, 1).date()
    end_date = datetime(year, 12, 31).date()
    
    # Cap at latest available disclosure (today - 60 days)
    latest_disclosure = (datetime.now() - timedelta(days=60)).date()
    if end_date > latest_disclosure:
        print(f"End date {end_date} exceeds 60-day lag. Capping at {latest_disclosure}.")
        end_date = latest_disclosure
        
    date_list = pd.date_range(start=start_date, end=end_date, freq='D')
    tasks = [(resource, d.date()) for d in date_list]
    
    print(f"Queueing {len(tasks)} days of data for {resource} ({start_date} to {end_date})...")
    
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_date = {executor.submit(fetch_worker, t): t[1] for t in tasks}
        
        for future in as_completed(future_to_date):
            date, status, points = future.result()
            results.append({'Date': date, 'Status': status, 'Points': points})
            # progress indicator
            print(f" Finished {date}: {status} ({points} pts)")
            
    df_results = pd.DataFrame(results).sort_values('Date')
    success_count = df_results[df_results['Status'] == 'Success'].shape[0]
    cached_count = df_results[df_results['Status'] == 'Cached'].shape[0]
    
    print("\n--- Summary ---")
    print(f"Total Days: {len(tasks)}")
    print(f"Successful: {success_count}")
    print(f"Cached: {cached_count}")
    print(f"Missing/Empty: {len(tasks) - success_count - cached_count}")
    
    print("\nData is now cached. You can run the benchmarking tool in app.py for this range.")

if __name__ == "__main__":
    main()
