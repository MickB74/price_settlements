#!/usr/bin/env python3
"""
Fetch missing SCED full disclosure files up to the 60-day lag limit.
Fills any gaps from the last cached date to today - 60 days.
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sced_fetcher

CACHE_DIR = sced_fetcher.CACHE_DIR

def main():
    # Latest available date (60-day lag)
    latest = (datetime.now() - timedelta(days=60)).date()

    # Find what's already cached
    existing = set()
    for f in os.listdir(CACHE_DIR):
        if f.startswith("full_disclosure_") and f.endswith(".parquet"):
            date_str = f.replace("full_disclosure_", "").replace(".parquet", "")
            try:
                existing.add(pd.Timestamp(date_str).date())
            except Exception:
                pass

    # Build list of missing dates (from 2024-01-01 to latest)
    start = datetime(2024, 1, 1).date()
    all_dates = pd.date_range(start=start, end=latest, freq="D")
    missing = [d.date() for d in all_dates if d.date() not in existing]

    print("=" * 60)
    print("SCED Full Disclosure Update")
    print("=" * 60)
    print(f"Latest available date (60-day lag): {latest}")
    print(f"Already cached: {len(existing)} days")
    print(f"Missing: {len(missing)} days")
    print("=" * 60)

    if not missing:
        print("\n✅ All disclosure files are up to date!")
        return

    success, failed = 0, 0
    for i, date in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] Fetching {date}...")
        try:
            df = sced_fetcher.get_daily_disclosure(date)
            if not df.empty:
                print(f"  ✓ {date}: {len(df):,} rows")
                success += 1
            else:
                print(f"  ✗ {date}: empty response")
                failed += 1
        except Exception as e:
            print(f"  ✗ {date}: ERROR - {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"✅ Done! Success: {success}, Failed/Empty: {failed}")
    print("=" * 60)

if __name__ == "__main__":
    main()
