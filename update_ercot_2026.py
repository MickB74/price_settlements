#!/usr/bin/env python3
"""
Update ercot_rtm_2026.parquet with all days in 2026.

This script:
1. Loads the existing parquet file (if any)
2. Fetches current 2026 data from ERCOT via gridstatus
3. Merges + deduplicates
4. Guards against accidental rollback (older max timestamp)
"""

import pandas as pd
import gridstatus
import patch_gridstatus  # Apply monkey patch for compatibility


def _load_existing(cache_file: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(cache_file)
        print(f"\n✓ Loaded existing file: {cache_file}")
        print(f"  - Total rows: {len(df):,}")
        if not df.empty and "Time_Central" in df.columns:
            print(f"  - Date range: {df['Time_Central'].min()} to {df['Time_Central'].max()}")
        return df
    except FileNotFoundError:
        print(f"\n! File not found: {cache_file}")
        print("  - Will create new file")
        return pd.DataFrame()


def update_ercot_2026():
    """Fetch and save complete 2026 ERCOT RTM data."""

    cache_file = "ercot_rtm_2026.parquet"

    print("=" * 60)
    print("ERCOT RTM 2026 Data Update Script")
    print("=" * 60)

    existing_df = _load_existing(cache_file)
    today = pd.Timestamp.now(tz='US/Central').date()

    start_date = pd.Timestamp("2026-01-01").date()
    existing_max = pd.NaT
    if not existing_df.empty and "Time_Central" in existing_df.columns:
        existing_max = pd.to_datetime(existing_df["Time_Central"], errors="coerce").max()
        if pd.notna(existing_max):
            # Re-fetch one prior day for corrections, but never before Jan 1.
            start_date = max(pd.Timestamp("2026-01-01").date(), (existing_max - pd.Timedelta(days=1)).date())

    if start_date > today:
        start_date = today

    print("\n" + "-" * 60)
    print(f"Fetching data from {start_date} to {today}...")
    print("-" * 60)

    iso = gridstatus.Ercot()

    try:
        new_df = pd.DataFrame()
        fetch_timestamp = pd.Timestamp.now(tz='US/Central')

        if existing_df.empty:
            print("Fetching base 2026 data (fast bulk)...")
            try:
                base_df = iso.get_rtm_spp(year=2026)
                if not base_df.empty:
                    # process base
                    if not pd.api.types.is_datetime64_any_dtype(base_df['Time']):
                        base_df['Time'] = pd.to_datetime(base_df['Time'], utc=True)
                    base_df['Time_Central'] = base_df['Time'].dt.tz_convert('US/Central')
                    base_df['fetched_at'] = fetch_timestamp
                    
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
                gap_df['fetched_at'] = fetch_timestamp
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

        # Deduplicate using latest pulled record for each Time+Location pair.
        print("Deduplicating...")
        combined = combined.sort_values('Time')
        combined = combined.drop_duplicates(subset=['Time', 'Location'], keep='last')
        
        # Ensure fetched_at column exists (backward compatibility)
        if 'fetched_at' not in combined.columns:
            combined['fetched_at'] = pd.NaT  # Set to NaT for old data
        
        # Memory Optimization
        float_cols = combined.select_dtypes(include=['float64']).columns
        for col in float_cols:
            combined[col] = pd.to_numeric(combined[col], downcast='float')
        
        # Add date column for easier filtering
        combined['date'] = combined['Time_Central'].dt.date

        combined_max = pd.to_datetime(combined["Time_Central"], errors="coerce").max()
        if pd.notna(existing_max) and pd.notna(combined_max) and combined_max < existing_max:
            # Never overwrite newer local data with older API responses.
            print("\n⚠️  Rollback guard triggered.")
            print(f"  Existing max timestamp: {existing_max}")
            print(f"  New combined max timestamp: {combined_max}")
            print("  Keeping existing local file unchanged.")
            return existing_df

        # Save
        print("\n" + "-" * 60)
        print(f"Saving updated data to {cache_file}...")
        print("-" * 60)
        
        combined.to_parquet(cache_file)
        
        print(f"\n✅ SUCCESS!")
        print(f"  - File updated: {cache_file}")
        print(f"  - Total rows: {len(combined):,}")
        print(f"  - Date range: {combined['Time_Central'].min()} to {combined['Time_Central'].max()}")
        if pd.notna(combined_max):
            lag_days = (pd.Timestamp.now(tz="US/Central") - combined_max).total_seconds() / 86400.0
            print(f"  - Market data lag vs now: {lag_days:.1f} days")

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
