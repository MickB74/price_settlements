import gridstatus
import pandas as pd

iso = gridstatus.Ercot()

try:
    print("Fetching Solar Regional Data for Jan 1, 2020...")
    # Checking method signature or just trying 'date' or 'year'
    # Most gridstatus methods take 'date', 'start', 'end'.
    # Let's try a specific date.
    df_solar = iso.get_solar_actual_and_forecast_by_geographical_region_hourly(date="2020-01-01")
    print("Solar Data Columns:", df_solar.columns)
    print(df_solar.head())
    
    print("\nFetching Wind Regional Data for Jan 1, 2020...")
    df_wind = iso.get_wind_actual_and_forecast_by_geographical_region_hourly(date="2020-01-01")
    print("Wind Data Columns:", df_wind.columns)
    print(df_wind.head())

except Exception as e:
    print(f"Error fetching regional data: {e}")
