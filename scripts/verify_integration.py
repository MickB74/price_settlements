import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fetch_tmy
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

def test_integration():
    print("Testing integration...")
    
    # Mock scenario
    year = 2024
    tech = "Solar"
    capacity = 100
    
    # 1. Fetch Profile
    print(f"Fetching {tech} profile for {year}...")
    profile = fetch_tmy.get_profile_for_year(year, tech, capacity)
    
    if profile.empty:
        print("ERROR: Profile is empty.")
        return
        
    print(f"Profile fetched. Length: {len(profile)}")
    print(profile.head())
    
    # 2. Mock RTM Data (Central Time)
    # Create a dummy RTM dataframe for a day
    dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-01-02", freq='15min', tz='US/Central')
    df_hub = pd.DataFrame({'Time_Central': dates})
    
    # 3. Align (Simulate app logic)
    print("Aligning to Central Time...")
    profile_central = profile.tz_convert('US/Central')
    potential_gen = profile_central.reindex(df_hub['Time_Central'], fill_value=0.0).values
    
    df_hub['Potential_Gen_MW'] = potential_gen
    
    print("Alignment successful.")
    print(df_hub.head())
    
    # Check for non-zero values (Solar should be 0 at night, >0 day)
    daytime = df_hub[df_hub['Time_Central'].dt.hour.between(10, 14)]
    if daytime['Potential_Gen_MW'].sum() > 0:
        print("SUCCESS: Daytime generation detected.")
    else:
        print("WARNING: No daytime generation.")

if __name__ == "__main__":
    test_integration()
