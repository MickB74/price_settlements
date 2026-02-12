import pandas as pd
import numpy as np
import os
import sys

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sced_fetcher
from datetime import date, timedelta

def get_2024_totals():
    resource = "AZURE_SKY_WIND_AGG"
    start = date(2024, 1, 1)
    end = date(2024, 12, 31)
    
    total_mwh_prior = 0
    total_mwh_twa = 0
    days_processed = 0
    
    current = start
    while current <= end:
        df_full = sced_fetcher.get_daily_disclosure(current)
        if not df_full.empty:
            # Aggregate Azure Sky
            v_units = ["VORTEX_WIND1", "VORTEX_WIND2", "VORTEX_WIND3", "VORTEX_WIND4"]
            df_asset = df_full[df_full['Resource Name'].isin(v_units)].copy()
            if not df_asset.empty:
                df_asset['Time'] = pd.to_datetime(df_asset['Interval Start'], utc=True)
                df_ts = df_asset.groupby('Time')['Telemetered Net Output'].sum().reset_index()
                
                # Prior
                df_ts['Interval_Start'] = df_ts['Time'].dt.floor('15min')
                total_mwh_prior += df_ts.groupby('Interval_Start')['Telemetered Net Output'].mean().sum() * 0.25
                
                # TWA
                # (Re-running TWA logic)
                start_bound = df_ts['Time'].min().floor('15min')
                end_bound = df_ts['Time'].max().ceil('15min')
                boundaries = pd.date_range(start=start_bound, end=end_bound, freq='15min', tz='UTC')
                all_ts = sorted(list(set(df_ts['Time'].tolist() + boundaries.tolist())))
                df_step = df_ts.set_index('Time')[['Telemetered Net Output']].reindex(all_ts).ffill().reset_index()
                df_step['next_time'] = df_step['Time'].shift(-1)
                df_step['duration_sec'] = (df_step['next_time'] - df_step['Time']).dt.total_seconds()
                df_step = df_step.dropna(subset=['next_time', 'duration_sec'])
                df_step = df_step[df_step['duration_sec'] > 0].copy()
                df_step['duration_sec'] = df_step['duration_sec'].clip(upper=3600)
                total_mwh_twa += (df_step['Telemetered Net Output'] * (df_step['duration_sec'] / 3600.0)).sum()
                
                days_processed += 1
                if days_processed % 30 == 0:
                    print(f"Processed {days_processed} days...")
                    
        current += timedelta(days=1)
        
    print(f"\n--- 2024 PERFORMANCE IMPACT (AZURE SKY WIND) ---")
    print(f"Total MWh (Prior): {total_mwh_prior:,.2f}")
    print(f"Total MWh (TWA):   {total_mwh_twa:,.2f}")
    print(f"Difference:       {total_mwh_twa - total_mwh_prior:,.2f} MWh")
    print(f"Percentage:       {(total_mwh_twa / total_mwh_prior - 1)*100:+.2f}%")

if __name__ == "__main__":
    get_2024_totals()
