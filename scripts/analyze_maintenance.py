
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fetch_tmy

def analyze_year(year):
    print(f"\n--- Analyzing {year} ---")
    
    import sced_fetcher
    from datetime import date
    
    # 1. Load Actuals
    cache_dir = "sced_cache"
    actual_file = os.path.join(cache_dir, f"AZURE_SKY_WIND_AGG_{year}_full.parquet")
    
    if not os.path.exists(actual_file):
        print(f"Cache file not found for {year}. Attempting to fetch...")
        try:
            start_date = date(year, 1, 1)
            end_date = date(year, 12, 31)
            # Cap at today
            if end_date > date.today():
                end_date = date.today()
                
            sced_fetcher.get_asset_period_data("AZURE_SKY_WIND_AGG", start_date, end_date)
            sced_fetcher.consolidate_year("AZURE_SKY_WIND_AGG", year)
        except Exception as e:
            print(f"Failed to fetch data: {e}")
            return
            
    if not os.path.exists(actual_file):
        print(f"Cache file still missing after fetch attempt: {actual_file}")
        return
        
    df_actual = pd.read_parquet(actual_file)
    df_actual = df_actual.rename(columns={"Actual_MW": "Actual"})
    if 'Time' in df_actual.columns:
        df_actual = df_actual.set_index("Time").sort_index()
    
    # Normalize index
    if df_actual.index.tz is None:
        df_actual.index = df_actual.index.tz_localize("UTC")
    else:
        df_actual.index = df_actual.index.tz_convert("UTC")

    # 2. Get Modeled Data (Mixed Fleet)
    lat = 33.1534
    lon = -99.2847
    turbines = [
        {'type': 'NORDEX_N149', 'count': 65, 'capacity_mw': 4.5, 'hub_height_m': 105.0},
        {'type': 'VESTAS_V163', 'count': 7,  'capacity_mw': 3.45, 'hub_height_m': 82.0},
        {'type': 'GENERIC',     'count': 7,  'capacity_mw': 2.0,  'hub_height_m': 80.0},
    ]
    
    print("Generating modeled profile...")
    try:
        s_modeled = fetch_tmy.get_blended_profile_for_year(
            year=year,
            tech="Wind",
            turbines=turbines,
            lat=lat,
            lon=lon
        )
    except Exception as e:
        print(f"Error generating model: {e}")
        return

    if s_modeled.empty:
        print("No modeled data.")
        return

    s_modeled.name = "Predicted"
    if s_modeled.index.tz is None:
        s_modeled.index = s_modeled.index.tz_localize("UTC")
    
    # 3. Merge
    df = pd.merge(df_actual[["Actual"]], s_modeled, left_index=True, right_index=True, how="inner")
    
    # 4. Analysis
    # Condition 1: Hard Downtime (Modeled > 20MW, Actual < 2MW)
    # 20MW is arbitrary but signifies "definite wind"
    # 2MW allows for parasitic load or minor calibration noise
    mask_downtime = (df["Predicted"] > 20) & (df["Actual"] < 2)
    downtime_hours = df[mask_downtime].index
    
    print(f"Total Intervals (15-min): {len(df)}")
    print(f"Potential Full Downtime Intervals: {len(downtime_hours)} ({len(downtime_hours)/4:.1f} hours)")
    
    if not downtime_hours.empty:
        # Group into consecutive periods
        df["is_downtime"] = mask_downtime.astype(int)
        df["group"] = (df["is_downtime"].diff() != 0).cumsum()
        
        events = []
        for g, d in df[mask_downtime].groupby("group"):
            start = d.index[0]
            end = d.index[-1]
            duration_hrs = (len(d) * 15) / 60
            lost_mwh = (d["Predicted"] * 0.25).sum() # approx
            
            if duration_hrs > 1.0: # Filter short blips
                events.append({
                    "start": start,
                    "end": end,
                    "duration_hours": duration_hrs,
                    "avg_modeled_mw": d["Predicted"].mean(),
                    "lost_mwh": lost_mwh
                })
        
        # Sort by lost MWh
        events.sort(key=lambda x: x["lost_mwh"], reverse=True)
        
        print("\nTop 5 Potential Maintenance Events (Full Outage):")
        for i, e in enumerate(events[:5]):
            print(f"{i+1}. {e['start'].strftime('%Y-%m-%d %H:%M')} ({e['duration_hours']:.1f} hrs) - Lost ~{e['lost_mwh']:.1f} MWh")

    # Condition 2: Significant Derating/Underperformance (Daily Level)
    # Resample to Daily
    daily = df.resample('D').sum() * 0.25 / 1000 # GWh
    daily = daily[daily["Predicted"] > 0.1] # Filter low wind days
    
    daily["Performance_Ratio"] = daily["Actual"] / daily["Predicted"]
    
    # Suspiciously low performance days (e.g. < 40% of modeled)
    low_perf_days = daily[daily["Performance_Ratio"] < 0.4].sort_values("Performance_Ratio")
    
    print(f"\nDays with < 40% Expected Generation: {len(low_perf_days)}")
    if not low_perf_days.empty:
        print("Top 5 Worst Performing Days (by Ratio):")
        print(low_perf_days[["Actual", "Predicted", "Performance_Ratio"]].head(5))

if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    analyze_year(year)
