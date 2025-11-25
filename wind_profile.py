import numpy as np
import pandas as pd

def generate_wind_profile(date_obj, peak_mw=80):
    """
    Generates a synthetic wind profile for a given date.
    
    Args:
        date_obj (datetime.date): The date to generate the profile for.
        peak_mw (float): The peak generation capacity in MW.
        
    Returns:
        pd.Series: A series of 96 values (15-min intervals) representing MW generation.
    """
    # Time intervals (0 to 23.75)
    intervals = np.arange(0, 24, 0.25)
    
    # Base pattern: Peak at 2 AM (Night), Trough at 2 PM (Day)
    # Sinusoidal wave: sin( (t - shift) * 2pi / 24 )
    # We want peak at t=2. 
    # sin( (2 - shift) * 2pi/24 ) = 1 => (2-shift)/12 * pi = pi/2 => (2-shift)/12 = 0.5 => 2-shift=6 => shift=-4
    
    # Base signal (-1 to 1)
    base_signal = np.sin((intervals + 4) * 2 * np.pi / 24)
    
    # Normalize to [0.2, 0.8] range to represent base capacity factor range (avoiding 0 and 100% constantly)
    # (base + 1) / 2 -> [0, 1]
    # Scale: 0.2 + 0.6 * normalized
    normalized_base = (base_signal + 1) / 2
    scaled_base = 0.2 + 0.6 * normalized_base
    
    # Add Randomness (Wind is stochastic)
    # Perlin-like noise would be better, but simple smoothed noise works for synthetic data
    np.random.seed(int(date_obj.strftime("%Y%m%d"))) # Seed with date for reproducibility
    noise = np.random.normal(0, 0.15, size=len(intervals))
    
    # Combine
    profile = scaled_base + noise
    
    # Clamp to [0, 1]
    profile = np.clip(profile, 0, 1)
    
    # Scale to MW
    mw_values = profile * peak_mw
    
    return pd.Series(mw_values, name="Wind_MW")

if __name__ == "__main__":
    from datetime import datetime
    test_date = datetime(2024, 1, 1).date()
    profile = generate_wind_profile(test_date)
    print(f"Wind Profile for {test_date} (Peak: {profile.max():.2f} MW, Mean: {profile.mean():.2f} MW):")
    print(profile.describe())
