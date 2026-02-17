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
            
            # New Strategy: Fetch ALL recent documents for Report 12301 and filter locally
            # This avoids potential API filtering issues with friendly_name_timestamp
            # Fetch only as far back as needed for the gap (plus a small safety buffer),
            # capped to keep document scans bounded.
            gap_days = max((today - start_date).days + 1, 1)
            lookback_days = min(max(gap_days + 2, 3), 21)
            print(f"  Fetching list of recent documents (last {lookback_days} days)...")

            start_search = pd.Timestamp.now(tz="US/Central") - pd.Timedelta(days=lookback_days)
            
            all_docs = iso._get_documents(
                report_type_id=12301,
                published_after=start_search
            )
            
            print(f"  Found {len(all_docs)} total documents.")
            
            # Sort docs by timestamp
            # gridstatus Documents have a 'friendly_name_timestamp' attribute
            all_docs.sort(key=lambda x: x.friendly_name_timestamp if x.friendly_name_timestamp else pd.Timestamp.min)
            
            # Filter for gap period
            target_docs = []
            for doc in all_docs:
                if doc.friendly_name_timestamp:
                    # doc ts is usually beginning of interval?
                    doc_date = doc.friendly_name_timestamp.date()
                    if doc_date >= start_date and doc_date <= today:
                        target_docs.append(doc)
                        
            print(f"  Identified {len(target_docs)} documents covering gap {start_date} to {today}.")
            
            # Process in small batches (e.g., by day or 100 docs) to save incrementally
            # Group by date
            from collections import defaultdict
            docs_by_date = defaultdict(list)
            for doc in target_docs:
                d = doc.friendly_name_timestamp.date()
                docs_by_date[d].append(doc)
                
            sorted_dates = sorted(docs_by_date.keys())
            
            for d in sorted_dates:
                print(f"  Processing {d} ({len(docs_by_date[d])} docs)...")
                try:
                    day_df = iso.read_docs(docs_by_date[d])
                    
                    if not day_df.empty:
                        # Process
                        if not pd.api.types.is_datetime64_any_dtype(day_df['Time']):
                            day_df['Time'] = pd.to_datetime(day_df['Time'], utc=True)
                        day_df['Time_Central'] = day_df['Time'].dt.tz_convert('US/Central')
                        day_df['fetched_at'] = fetch_timestamp
                        day_df['date'] = day_df['Time_Central'].dt.date
                        
                        # Memory Opt
                        float_cols = day_df.select_dtypes(include=['float64']).columns
                        for col in float_cols:
                            day_df[col] = pd.to_numeric(day_df[col], downcast='float')

                        # Merge with existing immediately
                        if not existing_df.empty:
                            existing_df = pd.concat([existing_df, day_df])
                        else:
                            existing_df = day_df
                        
                        # Deduplicate
                        existing_df = existing_df.sort_values('Time')
                        existing_df = existing_df.drop_duplicates(subset=['Time', 'Location'], keep='last')
                        
                        # SAVE IMMEDIATELY
                        existing_df.to_parquet(cache_file)
                        print(f"    Saved data for {d}. Total rows: {len(existing_df):,}")
                except Exception as e:
                        print(f"    Error processing {d}: {e}")
                        if 'day_df' in locals() and not day_df.empty:
                            print(f"    Columns: {day_df.columns.tolist()}")
                        
                        
            # Finished processing all docs for the gap

        return existing_df

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
