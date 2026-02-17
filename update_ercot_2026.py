#!/usr/bin/env python3
"""
Update ercot_rtm_2026.parquet with all days in 2026.

This script:
1. Loads the existing parquet file (if any)
2. Fetches current 2026 data from ERCOT via gridstatus
3. Merges + deduplicates
4. Guards against accidental rollback (older max timestamp)
"""

import sys
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
    """Fetch and save complete 2026 ERCOT RTM data in one deterministic pass."""

    cache_file = "ercot_rtm_2026.parquet"

    print("=" * 60)
    print("ERCOT RTM 2026 Data Update Script")
    print("=" * 60)

    existing_df = _load_existing(cache_file)
    existing_max = pd.NaT
    if not existing_df.empty and "Time_Central" in existing_df.columns:
        existing_max = pd.to_datetime(existing_df["Time_Central"], errors="coerce").max()

    print("\n" + "-" * 60)
    print("Fetching full 2026 ERCOT RTM dataset...")
    print("-" * 60)

    iso = gridstatus.Ercot()
    fetch_timestamp = pd.Timestamp.now(tz="US/Central")
    fresh_df = iso.get_rtm_spp(year=2026)
    if fresh_df is None or fresh_df.empty:
        raise RuntimeError("ERCOT returned no rows for 2026.")

    if not pd.api.types.is_datetime64_any_dtype(fresh_df["Time"]):
        fresh_df["Time"] = pd.to_datetime(fresh_df["Time"], utc=True, errors="coerce")
    fresh_df = fresh_df.dropna(subset=["Time"]).copy()
    fresh_df["Time_Central"] = fresh_df["Time"].dt.tz_convert("US/Central")
    fresh_df["fetched_at"] = fetch_timestamp
    fresh_df["date"] = fresh_df["Time_Central"].dt.date

    float_cols = fresh_df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        fresh_df[col] = pd.to_numeric(fresh_df[col], downcast="float")

    if existing_df.empty:
        combined = fresh_df
    else:
        combined = pd.concat([existing_df, fresh_df], ignore_index=True)
        combined = combined.sort_values("Time")
        combined = combined.drop_duplicates(subset=["Time", "Location"], keep="last")

    new_max = pd.to_datetime(combined["Time_Central"], errors="coerce").max()
    if pd.notna(existing_max) and pd.notna(new_max) and new_max < existing_max:
        raise RuntimeError(
            f"Rollback guard triggered. Existing max={existing_max}, new max={new_max}."
        )

    combined.to_parquet(cache_file)

    lag_days = (pd.Timestamp.now(tz="US/Central") - new_max).total_seconds() / 86400.0 if pd.notna(new_max) else float("nan")
    print(f"Saved {len(combined):,} rows to {cache_file}")
    print(f"Range: {combined['Time_Central'].min()} to {combined['Time_Central'].max()}")
    print(f"Lag vs now: {lag_days:.1f} days")
    return combined

if __name__ == "__main__":
    try:
        result = update_ercot_2026()
        print("\n" + "=" * 60)
        print("Update completed successfully! 🎉")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print("Update failed.")
        print(str(e))
        print("=" * 60)
        sys.exit(1)
