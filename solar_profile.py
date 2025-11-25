import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_solar_profile(date_obj, peak_mw=80, lat=31.0):
    """
    Generates a synthetic solar profile for a given date.
    
    Args:
        date_obj (datetime.date): The date to generate the profile for.
        peak_mw (float): The peak generation capacity in MW.
        lat (float): Latitude in degrees (approximate for Texas).
        
    Returns:
        pd.Series: A series of 96 values (15-min intervals) representing MW generation.
    """
    # Day of year (1-365)
    day_of_year = date_obj.timetuple().tm_yday
    
    # Approximate Day Length (hours)
    # Simple sinusoidal model: 12 + 2 * sin(...)
    # Longest day near day 172 (June 21), Shortest near day 355 (Dec 21)
    # Variation of +/- 2 hours is reasonable for 30-35 deg latitude
    day_length = 12 + 2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # Solar Noon is roughly 12:30 PM to 1:30 PM depending on longitude/DST
    # Let's assume Solar Noon is at 13:30 (1:30 PM) to account for DST/Timezone roughly
    solar_noon = 13.5
    
    sunrise = solar_noon - (day_length / 2)
    sunset = solar_noon + (day_length / 2)
    
    # Generate 15-minute intervals (0 to 24 hours)
    intervals = np.arange(0, 24, 0.25) # 0.0, 0.25, 0.5, ... 23.75
    
    generation = []
    for t in intervals:
        if t < sunrise or t > sunset:
            mw = 0.0
        else:
            # Sinusoidal shape during the day
            # Normalize time within the daylight window to [0, pi]
            # (t - sunrise) / day_length goes from 0 to 1
            # Multiply by pi to go from 0 to pi
            mw = peak_mw * np.sin(np.pi * (t - sunrise) / day_length)
        generation.append(max(0.0, mw))
        
    return pd.Series(generation, name="Solar_MW")

if __name__ == "__main__":
    # Test
    test_date = datetime(2024, 6, 21).date()
    profile = generate_solar_profile(test_date)
    print(f"Profile for {test_date} (Peak: {profile.max():.2f} MW):")
    print(profile.describe())
