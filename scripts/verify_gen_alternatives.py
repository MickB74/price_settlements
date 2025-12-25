import gridstatus
import pandas as pd

iso = gridstatus.Ercot()

print("--- Test 1: System-Wide Solar for 2020 ---")
try:
    # System wide might be more available
    df_sys = iso.get_solar_actual_and_forecast_hourly(date="2020-01-01", end="2020-01-02")
    print("System Wide 2020 Columns:", df_sys.columns)
    print(df_sys.head())
except Exception as e:
    print(f"System Wide 2020 Failed: {e}")

print("\n--- Test 2: Regional Solar for 2023 ---")
try:
    # Try a more recent date
    df_reg = iso.get_solar_actual_and_forecast_by_geographical_region_hourly(date="2023-01-01", end="2023-01-02")
    print("Regional 2023 Columns:", df_reg.columns)
    print(df_reg.head())
except Exception as e:
    print(f"Regional 2023 Failed: {e}")
