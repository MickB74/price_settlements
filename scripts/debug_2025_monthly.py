import pandas as pd
import os
import sys

filename = "ercot_rtm_2025.parquet"
df = pd.read_parquet(filename)

# Ensure Time_Central
if 'Time_Central' not in df.columns:
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    df['Time_Central'] = df['Time'].dt.tz_convert('US/Central')

# Use 'Location' based on previous output
loc_col = 'Location'
price_col = 'SPP'

print(f"Filtering {loc_col} == 'HB_SOUTH'...")
hub_data = df[df[loc_col] == 'HB_SOUTH'].copy()
print(f"HB_SOUTH Rows: {len(hub_data)}")

# Monthly Stats
hub_data['Month'] = hub_data['Time_Central'].dt.month
monthly = hub_data.groupby('Month')[price_col].agg(['mean', 'min', 'max', 'count'])

print("\n--- Monthly Price Stats (HB_SOUTH) ---")
print(monthly)

print("\n--- Overall ---")
print(f"Mean Price: ${hub_data[price_col].mean():.2f}")
