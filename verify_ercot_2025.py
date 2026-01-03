#!/usr/bin/env python3
"""Quick verification of the updated ercot_rtm_2025.parquet file."""

import pandas as pd
import os

# Load the file
df = pd.read_parquet('ercot_rtm_2025.parquet')

print("=" * 60)
print("ERCOT RTM 2025 File Verification")
print("=" * 60)

print(f"\n✓ File loaded successfully")
print(f"  Total rows: {len(df):,}")
print(f"  Date range: {df['Time_Central'].min()} to {df['Time_Central'].max()}")
print(f"  File size: {round(os.path.getsize('ercot_rtm_2025.parquet') / 1024 / 1024, 2)} MB on disk")

# Check hub data
print(f"\n✓ Hub data (HB_* locations):")
hubs = df[df['Location'].str.startswith('HB_', na=False)]
print(f"  Total hub records: {len(hubs):,}")

print(f"\n  Records by hub:")
for hub in sorted(hubs['Location'].unique()):
    count = len(hubs[hubs['Location'] == hub])
    print(f"    {hub}: {count:,} records")

# Verify we have all days
print(f"\n✓ Date coverage:")
days_in_data = df['Time_Central'].dt.date.nunique()
print(f"  Unique days: {days_in_data}")
print(f"  Expected days in 2025: 365")

if days_in_data == 365:
    print(f"  ✅ Complete year data!")
else:
    print(f"  ⚠️  Missing {365 - days_in_data} days")

print("\n" + "=" * 60)
