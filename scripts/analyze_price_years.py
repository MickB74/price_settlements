"""
Check if Monte Carlo's -$10M P50 is correct by analyzing price year distribution.
Compare 2025 prices vs 2020-2024 average.
"""
import pandas as pd

years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
hub = 'HB_SOUTH'

results = []
for year in years:
    try:
        df = pd.read_parquet(f'ercot_rtm_{year}.parquet')
        df_hub = df[df['Location'] == hub]
        
        if df_hub.empty:
            print(f"{year}: No data")
            continue
            
        mean_price = df_hub['SPP'].mean()
        results.append({
            'year': year,
            'mean_price': mean_price,
            'settlement_per_mwh': mean_price - 50.0  # vs $50 strike
        })
        print(f"{year}: Mean ${mean_price:.2f}, Settlement $/MWh: ${mean_price - 50:.2f}")
    except Exception as e:
        print(f"{year}: Error - {e}")

if results:
    df_results = pd.DataFrame(results)
    print(f"\n=== SUMMARY ===")
    print(f"Average settlement $/MWh (2020-2026): ${df_results['settlement_per_mwh'].mean():.2f}")
    print(f"2025 settlement $/MWh: ${df_results[df_results['year']==2025]['settlement_per_mwh'].values[0]:.2f}")
    
    # Estimate annual settlement assuming 200k MWh/year
    avg_annual = df_results['settlement_per_mwh'].mean() * 200000
    print(f"\nEstimated MC P50 (200k MWh): ${avg_annual:,.0f}")
