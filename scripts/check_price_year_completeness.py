"""
Check how many rows each price year has.
Full year should have 35,040 rows (15-min intervals).
"""
import pandas as pd

years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
hub = 'HB_SOUTH'

print("Price Year Completeness Check:")
print("=" * 60)
for year in years:
    try:
        df = pd.read_parquet(f'ercot_rtm_{year}.parquet')
        df_hub = df[df['Location'] == hub]
        
        expected = 35040  # Full year
        pct = (len(df_hub) / expected) * 100
        
        status = "✅ COMPLETE" if pct >= 99 else f"⚠️  PARTIAL ({pct:.1f}%)"
        print(f"{year}: {len(df_hub):,} rows - {status}")
        
    except Exception as e:
        print(f"{year}: ERROR - {e}")

print("\n" + "=" * 60)
print("RECOMMENDATION: Exclude years with <99% data from Monte Carlo")
