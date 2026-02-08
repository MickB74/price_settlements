import pandas as pd
import os
import sys

filename = "ercot_rtm_2025.parquet"

if not os.path.exists(filename):
    print(f"File {filename} not found.")
    # Try 2024 to compare if 2025 missing
    filename = "ercot_rtm_2024.parquet"
    if not os.path.exists(filename):
        print("No RTM data found.")
        sys.exit(1)

print(f"Loading {filename}...")
try:
    df = pd.read_parquet(filename)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

if 'Time' in df.columns:
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    if 'Time_Central' not in df.columns:
        df['Time_Central'] = df['Time'].dt.tz_convert('US/Central')
    
    # Check Interval
    df = df.sort_values('Time')
    diff = df['Time'].diff().dropna()
    print("\nTime Interval Stats:")
    print(diff.value_counts())
    
    # Check range
    print(f"\nStart: {df['Time_Central'].min()}")
    print(f"End:   {df['Time_Central'].max()}")
    duration = df['Time_Central'].max() - df['Time_Central'].min()
    print(f"Duration: {duration}")

if 'Settlement Point Price' in df.columns:
    price_col = 'Settlement Point Price'
elif 'SPP' in df.columns:
    price_col = 'SPP'
else:
    # Try looking for float cols
    price_col = [c for c in df.columns if 'Price' in c][0]

print(f"\nPrice Column: {price_col}")
# Filter for South Hub if possible
if 'Settlement Point Name' in df.columns:
    hub_name = 'HB_SOUTH'
    df_hub = df[df['Settlement Point Name'] == hub_name]
    print(f"\nStats for {hub_name}:")
    print(df_hub[price_col].describe())
    
    # Check for Zeros
    zeros = (df_hub[price_col] == 0).sum()
    print(f"Zero Prices: {zeros} ({zeros/len(df_hub):.1%})")
    
    # Check for negatives
    neg = (df_hub[price_col] < 0).sum()
    print(f"Negative Prices: {neg} ({neg/len(df_hub):.1%})")

else:
    print(f"\nStats for All Locations:")
    print(df[price_col].describe())
