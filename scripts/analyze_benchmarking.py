import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sced_fetcher
import fetch_tmy

def analyze(resource_name, year, tech, capacity_mw, lat, lon):
    print(f"--- Analyzing {resource_name} for {year} ---")
    
    # 1. Load Actuals (Cached)
    print("Loading Actual SCED Data...")
    start_date = datetime(year, 1, 1).date()
    end_date = datetime(year, 12, 31).date()
    df_actual = sced_fetcher.get_asset_period_data(resource_name, start_date, end_date)
    
    if df_actual.empty:
        print("No actual data found!")
        return

    print(f"Loaded {len(df_actual)} actual data points.")
    
    # 2. Run Model
    print("Generating Model Profile (ERA5)...")
    s_modeled = fetch_tmy.get_profile_for_year(year, tech, capacity_mw, lat, lon)
    
    # 3. Align
    df_modeled = s_modeled.to_frame(name='Modeled_MW')
    df_modeled.index = pd.to_datetime(df_modeled.index, utc=True)
    
    # Merge on Time
    df_comp = pd.merge(df_actual, df_modeled, left_on='Time', right_index=True, how='inner')
    
    print(f"Aligned {len(df_comp)} common timestamps.")
    
    # 4. Metrics
    total_actual_mwh = df_comp['Actual_MW'].sum() / 4
    total_modeled_mwh = df_comp['Modeled_MW'].sum() / 4
    
    diff_mwh = total_modeled_mwh - total_actual_mwh
    pct_diff = (diff_mwh / total_actual_mwh) * 100
    
    mae = (df_comp['Actual_MW'] - df_comp['Modeled_MW']).abs().mean()
    rmse = np.sqrt(((df_comp['Actual_MW'] - df_comp['Modeled_MW']) ** 2).mean())
    correlation = df_comp['Actual_MW'].corr(df_comp['Modeled_MW'])
    r2 = correlation ** 2
    
    # 5. Report
    print("\n=== PERFORMANCE RESULTS ===")
    print(f"Total Actual Energy:  {total_actual_mwh:,.2f} MWh")
    print(f"Total Modeled Energy: {total_modeled_mwh:,.2f} MWh")
    print(f"Difference:           {diff_mwh:,.2f} MWh ({pct_diff:+.2f}%)")
    print("---------------------------")
    print(f"Correlation (R):      {correlation:.4f}")
    print(f"R-Squared (R²):       {r2:.4f}")
    print(f"Mean Abs Error (MAE): {mae:.2f} MW")
    print(f"RMSE:                 {rmse:.2f} MW")
    print("===========================")
    
    # Monthly Breakdown
    df_comp['Month'] = df_comp['Time'].dt.month
    print("\n--- Monthly Accuracy (Correlation) ---")
    monthly_corr = df_comp.groupby('Month').apply(lambda x: x['Actual_MW'].corr(x['Modeled_MW']))
    print(monthly_corr)

if __name__ == "__main__":
    # Frye Solar Specs
    # Capacity: 200 MW (from ercot_assets.json)
    # Location: Swisher County (Lat: 34.4067, Lon: -101.5966)
    analyze("FRYE_SLR_UNIT1", 2024, "Solar", 200.0, 34.4067, -101.5966)
