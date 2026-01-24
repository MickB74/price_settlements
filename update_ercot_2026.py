#!/usr/bin/env python3
"""
Update ercot_rtm_2026.parquet with all days in 2026.

This script:
1. Loads the existing parquet file (if any)
2. Fetches the complete 2026 data from ERCOT via gridstatus
3. Saves the updated file with all days in 2026
"""

import pandas as pd
import gridstatus
import patch_gridstatus  # Apply monkey patch for compatibility

def update_ercot_2026():
    """Fetch and save complete 2026 ERCOT RTM data."""
    
    cache_file = "ercot_rtm_2026.parquet"
    
    print("=" * 60)
    print("ERCOT RTM 2026 Data Update Script")
    print("=" * 60)
    
    # Check existing file
    try:
        existing_df = pd.read_parquet(cache_file)
        print(f"\n✓ Loaded existing file: {cache_file}")
        print(f"  - Total rows: {len(existing_df):,}")
        print(f"  - Date range: {existing_df['Time'].min()} to {existing_df['Time'].max()}")
    except FileNotFoundError:
        print(f"\n! File not found: {cache_file}")
        print("  - Will create new file")
        existing_df = None
    
    today = pd.Timestamp.now(tz='US/Central').date()
    
    # Check existing file
    start_date = pd.Timestamp("2026-01-01").date()
    # default behavior
    existing_df = pd.DataFrame()

    try:
        existing_df = pd.read_parquet(cache_file)
        print(f"\n✓ Loaded existing file: {cache_file}")
        print(f"  - Total rows: {len(existing_df):,}")
        if not existing_df.empty:
            max_dt = existing_df['Time_Central'].max()
            print(f"  - Date range: {existing_df['Time_Central'].min()} to {max_dt}")
            # Start from the next day after the last data point
            start_date = max_dt.date()
        else:
             print("  - File is empty (will fetch full year)")
             
    except FileNotFoundError:
        print(f"\n! File not found: {cache_file}")
        print("  - Will fetch starting from Jan 1, 2026")
        existing_df = pd.DataFrame()

    # Determine fetch range
    # Ensure we don't fetch future if local file is up to date (though RTM is fast)
    if start_date >= today:
        print("  - Data appears up to date. Checking for intraday updates...")
        start_date = today # Check today just in case

    print("\n" + "-" * 60)
    print(f"Fetching data from {start_date} to {today}...")
    print("-" * 60)
    
    iso = gridstatus.Ercot()
    
    try:
        # Strategy:
        # 1. If we have a big gap (more than a few days) or no data, maybe fetch full year first if efficient?
        #    Actually get_rtm_spp(year=2026) is fast for bulk but stale.
        #    Let's stick to get_spp for the gap if we have existing data.
        #    If no existing data, maybe get_rtm_spp(2026) first then gap fill?
        
        new_df = pd.DataFrame()
        
        if existing_df.empty:
            print("Fetching base 2026 data (fast bulk)...")
            try:
                base_df = iso.get_rtm_spp(year=2026)
                if not base_df.empty:
                    # process base
                    if not pd.api.types.is_datetime64_any_dtype(base_df['Time']):
                        base_df['Time'] = pd.to_datetime(base_df['Time'], utc=True)
                    base_df['Time_Central'] = base_df['Time'].dt.tz_convert('US/Central')
                    
                    existing_df = base_df
                    # Update start_date based on what we just got
                    start_date = existing_df['Time_Central'].max().date()
                    print(f"Fetched base data up to {start_date}")
            except Exception as e:
                print(f"Bulk fetch failed/empty: {e}")

        # Now fetch the gap (or refinement)
        if start_date <= today:
            print(f"Fetching detailed interval data from {start_date} to {today}...")
            # get_spp supports range
            gap_df = iso.get_spp(date=start_date, end=today, market="REAL_TIME_15_MIN")

            if not gap_df.empty:
                print(f"Fetched {len(gap_df)} recent rows")
                # Pre-process gap_df
                if not pd.api.types.is_datetime64_any_dtype(gap_df['Time']):
                    gap_df['Time'] = pd.to_datetime(gap_df['Time'], utc=True)
                gap_df['Time_Central'] = gap_df['Time'].dt.tz_convert('US/Central')
                new_df = gap_df

        # Merge
        if not new_df.empty:
            if not existing_df.empty:
                combined = pd.concat([existing_df, new_df])
            else:
                combined = new_df
        else:
            combined = existing_df

        if combined.empty:
             print("No data found.")
             return None

        # Deduplicate
        # Sort by time and keep last (assuming newer fetch might correct older?)
        # Or usually just drop duplicates based on Time + Location
        print("Deduplicating...")
        combined = combined.sort_values('Time')
        combined = combined.drop_duplicates(subset=['Time', 'Location'], keep='last')
        
        # Memory Optimization
        float_cols = combined.select_dtypes(include=['float64']).columns
        for col in float_cols:
            combined[col] = pd.to_numeric(combined[col], downcast='float')
        
        # Save
        print("\n" + "-" * 60)
        print(f"Saving updated data to {cache_file}...")
        print("-" * 60)
        
        combined.to_parquet(cache_file)
        
        print(f"\n✅ SUCCESS!")
        print(f"  - File updated: {cache_file}")
        print(f"  - Total rows: {len(combined):,}")
        print(f"  - Date range: {combined['Time_Central'].min()} to {combined['Time_Central'].max()}")
        
        return combined
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to update data")
        print(f"  {str(e)}")
        # If we have existing data, we survived but didn't update
        return existing_df

if __name__ == "__main__":
    result = update_ercot_2026()
    
    if result is not None:
        print("\n" + "=" * 60)
        print("Update completed successfully! 🎉")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Update failed. Please check error messages above.")
        print("=" * 60)
