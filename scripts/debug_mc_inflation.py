"""
Debug script to compare single-year calculation vs Monte Carlo for identical inputs.
This will help identify why MC P50 is 2.3x higher than single-year result.
"""
import pandas as pd
import sys
sys.path.append('.')

from utils import monte_carlo
import fetch_tmy

# Test config matching user's scenario
config = {
    'hub': 'HB_SOUTH',
    'tech': 'Solar',
    'capacity_mw': 80.0,
    'lat': 26.9070,  # HB_SOUTH coordinates
    'lon': -99.2715,
    'vppa_price': 50.0,
    'revenue_share': 100,
    'curtail_neg': False
}

# Load 2025 price data
print("Loading 2025 price data...")
df_price = pd.read_parquet('ercot_rtm_2025.parquet')
print(f"Total price rows: {len(df_price)}")

# Filter by hub
df_hub = df_price[df_price['Location'] == 'HB_SOUTH'].copy()
print(f"HB_SOUTH price rows: {len(df_hub)}")
print(f"Unique timestamps: {df_hub['Time_Central'].nunique()}")

# Load 2025 generation profile
print("\nLoading 2025 Solar profile...")
gen_profile = fetch_tmy.get_profile_for_year(
    year=2025,
    tech='Solar',
    capacity_mw=80.0,
    lat=26.9070,
    lon=-99.2715
)
print(f"Generation profile length: {len(gen_profile)}")

# Convert to Central
gen_central = gen_profile.tz_convert('US/Central')

# Create gen_df (Monte Carlo style)
gen_df = pd.DataFrame({
    'Gen_MW': gen_central.values,
    'Time_Central': gen_central.index
})
gen_df['Gen_Energy_MWh'] = gen_df['Gen_MW'] * 0.25
print(f"\ngen_df rows: {len(gen_df)}")
print(f"gen_df unique timestamps: {gen_df['Time_Central'].nunique()}")

# Check for duplicates
dups = gen_df[gen_df.duplicated(subset=['Time_Central'], keep=False)]
if not dups.empty:
    print(f"\n⚠️  DUPLICATES FOUND in gen_df: {len(dups)} rows")
    print(dups.head(20))

# Merge (Monte Carlo style)
merged = pd.merge(
    df_hub,
    gen_df[['Time_Central', 'Gen_Energy_MWh', 'Gen_MW']],
    on='Time_Central',
    how='inner'
)
print(f"\nMerged rows: {len(merged)}")
print(f"Merged unique timestamps: {merged['Time_Central'].nunique()}")

# Check for duplicates in merged
merged_dups = merged[merged.duplicated(subset=['Time_Central'], keep=False)]
if not merged_dups.empty:
    print(f"\n⚠️  DUPLICATES FOUND in merged: {len(merged_dups)} rows")
    print(merged_dups[['Time_Central', 'SPP', 'Gen_MW', 'Gen_Energy_MWh']].head(20))

# Calculate settlement
price_diff = merged['SPP'] - 50.0
merged['Settlement_$'] = merged['Gen_Energy_MWh'] * price_diff

annual_settlement = merged['Settlement_$'].sum()
annual_generation = merged['Gen_Energy_MWh'].sum()

print(f"\n=== RESULTS ===")
print(f"Annual Settlement: ${annual_settlement:,.2f}")
print(f"Annual Generation: {annual_generation:,.0f} MWh")
print(f"Avg Settlement $/MWh: ${annual_settlement/annual_generation:.2f}")

# Compare to expected
expected = -4393381.02
ratio = annual_settlement / expected
print(f"\nExpected: ${expected:,.2f}")
print(f"Ratio: {ratio:.2f}x")
