"""
Check if price_df has duplicate timestamps even after filtering by Location.
This could explain the 5x inflation.
"""
import pandas as pd

df = pd.read_parquet('ercot_rtm_2025.parquet')
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nUnique values per column:")
for col in df.columns:
    if col not in ['Time', 'Time_Central', 'Interval Start', 'Interval End', 'SPP']:
        nunique = df[col].nunique()
        print(f"  {col}: {nunique} unique values")
        if nunique < 20:
            print(f"    Values: {sorted(df[col].unique())}")

# Filter by HB_SOUTH
df_hub = df[df['Location'] == 'HB_SOUTH'].copy()
print(f"\nAfter filtering Location==HB_SOUTH:")
print(f"  Rows: {len(df_hub)}")
print(f"  Unique timestamps: {df_hub['Time_Central'].nunique()}")

# Check for duplicates
dups = df_hub[df_hub.duplicated(subset=['Time_Central'], keep=False)]
if not dups.empty:
    print(f"\n⚠️  DUPLICATE TIMESTAMPS FOUND: {len(dups)} rows")
    print("\nSample duplicates:")
    print(dups[['Time_Central', 'Location', 'Location Type', 'Market', 'SPP']].head(20))
    
    # Count duplicates per timestamp
    dup_counts = dups.groupby('Time_Central').size()
    print(f"\nDuplication factor: {dup_counts.mean():.1f}x average")
    print(f"Max duplicates for single timestamp: {dup_counts.max()}")
else:
    print("\n✅ No duplicate timestamps found")
