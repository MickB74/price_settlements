"""
Calculate what the P50 SHOULD be based on the 7 price years.
"""
import pandas as pd
import numpy as np

# From previous analysis
price_years = {
    2020: -28.54,
    2021: 96.68,
    2022: 10.56,
    2023: -3.90,
    2024: -22.93,
    2025: -17.98,
    2026: -9.96
}

settlements_per_mwh = list(price_years.values())
settlements_per_mwh.sort()

print("Sorted settlement $/MWh by price year:")
for i, val in enumerate(settlements_per_mwh):
    print(f"  {i+1}. ${val:.2f}")

median_settlement_per_mwh = np.median(settlements_per_mwh)
print(f"\nMedian settlement $/MWh: ${median_settlement_per_mwh:.2f}")

# Assuming 200k MWh/year
annual_p50 = median_settlement_per_mwh * 200000
print(f"Expected MC P50 (200k MWh): ${annual_p50:,.0f}")

# But user's actual single-year 2025 result was -$4.39M for 195k MWh
# Let's recalculate with actual generation
actual_mwh = 195493  # From user's screenshot
annual_p50_adjusted = median_settlement_per_mwh * actual_mwh
print(f"\nExpected MC P50 (195k MWh): ${annual_p50_adjusted:,.0f}")

# User reported MC P50 of -$10.2M
reported_mc_p50 = -10_200_000
ratio = reported_mc_p50 / annual_p50_adjusted
print(f"\nReported MC P50: ${reported_mc_p50:,.0f}")
print(f"Ratio vs expected: {ratio:.2f}x")
