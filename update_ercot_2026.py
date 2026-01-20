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
    
    # Fetch complete 2026 data
    print("\n" + "-" * 60)
    print("Fetching complete 2026 data from ERCOT...")
    print("⏳ This may take 1-2 minutes...")
    print("-" * 60)
    
    iso = gridstatus.Ercot()
    
    try:
        df = iso.get_rtm_spp(year=2026)
        
        print(f"\n✓ Successfully fetched 2026 data")
        print(f"  - Total rows: {len(df):,}")
        print(f"  - Date range: {df['Time'].min()} to {df['Time'].max()}")
        
        # Pre-process: Ensure Time is datetime and localized
        if not pd.api.types.is_datetime64_any_dtype(df['Time']):
            df['Time'] = pd.to_datetime(df['Time'], utc=True)
        
        # Create Central Time column
        df['Time_Central'] = df['Time'].dt.tz_convert('US/Central')
        
        # Memory Optimization: Downcast float64 to float32
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Save to parquet
        print("\n" + "-" * 60)
        print(f"Saving updated data to {cache_file}...")
        print("-" * 60)
        
        df.to_parquet(cache_file)
        
        print(f"\n✅ SUCCESS!")
        print(f"  - File updated: {cache_file}")
        print(f"  - Total rows: {len(df):,}")
        print(f"  - Date range: {df['Time_Central'].min()} to {df['Time_Central'].max()}")
        
        # Show data by location (hubs)
        print("\n" + "-" * 60)
        print("Data summary by location:")
        print("-" * 60)
        location_counts = df['Location'].value_counts()
        for loc, count in location_counts.head(10).items():
            print(f"  {loc}: {count:,} records")
        
        return df
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to fetch data")
        print(f"  {str(e)}")
        return None

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
