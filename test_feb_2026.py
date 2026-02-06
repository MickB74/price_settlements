#!/usr/bin/env python3
"""
Quick diagnostic to test if February 2026 data is accessible.
Run this to verify the data is properly loaded.
"""

import pandas as pd

print("=" * 60)
print("FEBRUARY 2026 DATA DIAGNOSTIC")
print("=" * 60)
print()

# Test 1: Can we load the file?
try:
    df = pd.read_parquet('ercot_rtm_2026.parquet')
    print("✅ Successfully loaded ercot_rtm_2026.parquet")
    print(f"   Total rows: {len(df):,}")
    print(f"   Date range: {df['Time_Central'].min()} to {df['Time_Central'].max()}")
except Exception as e:
    print(f"❌ Failed to load file: {e}")
    exit(1)

print()

# Test 2: Is there February data?
df_feb = df[df['Time_Central'].dt.month == 2]
print(f"February 2026 data: {len(df_feb):,} rows")

if len(df_feb) == 0:
    print("❌ NO FEBRUARY DATA FOUND!")
    exit(1)

print(f"✅ February data exists: {df_feb['Time_Central'].min()} to {df_feb['Time_Central'].max()}")
print()

# Test 3: Check each hub has February data
print("Checking hubs...")
hubs = ['HB_NORTH', 'HB_SOUTH', 'HB_WEST', 'HB_HOUSTON', 'HB_PAN']

for hub in hubs:
    hub_feb = df_feb[df_feb['Location'] == hub]
    if len(hub_feb) > 0:
        print(f"  ✅ {hub:12} {len(hub_feb):,} rows")
    else:
        print(f"  ❌ {hub:12} NO DATA")

print()
print("=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
print()
print("If all checks passed, the data is fine.")
print("The issue is likely with the app configuration.")
print()
print("In the Streamlit app sidebar:")
print("1. ✅ Check the 'Filter by specific month' checkbox")
print("2. ✅ Select 'February' from the month dropdown")  
print("3. ✅ Make sure year is set to 2026")
print("4. ✅ Click 'Add Scenario' then 'Run Scenarios'")
