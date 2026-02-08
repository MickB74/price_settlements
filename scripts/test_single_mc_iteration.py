"""
Manually simulate ONE Monte Carlo iteration to find the 5x bug.
Use weather_year=2025, price_year=2026 (median price year).
Expected result: -$9.96/MWh * 195k MWh = -$1.94M
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('.')
import fetch_tmy

# Config
config = {
    'hub': 'HB_SOUTH',
    'tech': 'Solar',
    'capacity_mw': 80.0,
    'lat': 26.9070,
    'lon': -99.2715,
    'vppa_price': 50.0,
    'revenue_share': 100,
    'curtail_neg': False
}

weather_year = 2025
price_year = 2026

print(f"=== Manual MC Iteration: weather={weather_year}, price={price_year} ===\n")

# 1. Load generation profile
print("1. Loading generation profile...")
gen_profile = fetch_tmy.get_profile_for_year(
    year=weather_year,
    tech='Solar',
    capacity_mw=80.0,
    lat=26.9070,
    lon=-99.2715,
    force_tmy=False
)
print(f"   Profile length: {len(gen_profile)}")
print(f"   Mean MW: {gen_profile.mean():.2f}")

# 2. Convert to Central and create gen_df
gen_central = gen_profile.tz_convert('US/Central')
gen_df = pd.DataFrame({
    'Gen_MW': gen_central.values,
    'Time_Source': gen_central.index
})

# 3. Shift timestamps to price year (Monte Carlo logic)
def replace_year(ts):
    try:
        return ts.replace(year=price_year)
    except ValueError:
        return ts + pd.DateOffset(days=1)

gen_df['Time_Central'] = gen_df['Time_Source'].apply(replace_year)
gen_df['Gen_Energy_MWh'] = gen_df['Gen_MW'] * 0.25

print(f"\n2. Created gen_df:")
print(f"   Rows: {len(gen_df)}")
print(f"   Total MWh: {gen_df['Gen_Energy_MWh'].sum():,.0f}")

# 4. Load price data
print(f"\n3. Loading price data for {price_year}...")
df_price = pd.read_parquet(f'ercot_rtm_{price_year}.parquet')
df_hub = df_price[df_price['Location'] == 'HB_SOUTH'].copy()
print(f"   Price rows: {len(df_hub)}")
print(f"   Mean price: ${df_hub['SPP'].mean():.2f}")

# 5. Merge
print(f"\n4. Merging...")
merged = pd.merge(
    df_hub,
    gen_df[['Time_Central', 'Gen_Energy_MWh', 'Gen_MW']],
    on='Time_Central',
    how='inner'
)
print(f"   Merged rows: {len(merged)}")
print(f"   Unique timestamps: {merged['Time_Central'].nunique()}")

# Check for duplicates
dups = merged[merged.duplicated(subset=['Time_Central'], keep=False)]
if not dups.empty:
    print(f"   ⚠️  DUPLICATES: {len(dups)} rows!")
    print(dups[['Time_Central', 'SPP', 'Gen_MW']].head())

# 6. Calculate settlement
price_diff = merged['SPP'] - 50.0
merged['Settlement_$'] = merged['Gen_Energy_MWh'] * price_diff

annual_settlement = merged['Settlement_$'].sum()
annual_generation = merged['Gen_Energy_MWh'].sum()

print(f"\n=== RESULTS ===")
print(f"Annual Settlement: ${annual_settlement:,.2f}")
print(f"Annual Generation: {annual_generation:,.0f} MWh")
print(f"Settlement $/MWh: ${annual_settlement/annual_generation:.2f}")

# Expected
expected_per_mwh = df_hub['SPP'].mean() - 50.0
expected_annual = expected_per_mwh * annual_generation
print(f"\nExpected (based on mean price):")
print(f"  Settlement $/MWh: ${expected_per_mwh:.2f}")
print(f"  Annual: ${expected_annual:,.2f}")
print(f"\nRatio (actual/expected): {annual_settlement/expected_annual:.2f}x")
