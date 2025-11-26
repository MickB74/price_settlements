import gridstatus
import pandas as pd

iso = gridstatus.Ercot()
print("Methods in gridstatus.Ercot:")
for method in dir(iso):
    if not method.startswith("_"):
        print(method)

# Try to fetch fuel mix or generation
try:
    print("\nAttempting to fetch fuel mix (often contains wind/solar)...")
    # Fuel mix usually gives system-wide generation by type
    df_mix = iso.get_fuel_mix(date="today")
    print(df_mix.head())
    print("Columns:", df_mix.columns)
except Exception as e:
    print(f"Error fetching fuel mix: {e}")

# Check for other potential methods like 'get_hourly_wind_solar' or similar
# (Guessing names, but dir() will show actuals)
