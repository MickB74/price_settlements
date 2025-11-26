import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Cache directory
CACHE_DIR = "data_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_tmy_data(lat=32.4487, lon=-99.7331, force_refresh=False):
    """
    Fetches TMY data from PVGIS for the given coordinates.
    Defaults to Abilene, Texas (approximate center for West Texas wind/solar).
    
    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        force_refresh (bool): If True, ignores cache and fetches fresh data.
        
    Returns:
        pd.DataFrame: DataFrame with TMY data (Solar GHI, Wind Speed, etc.)
    """
    cache_file = os.path.join(CACHE_DIR, f"tmy_{lat}_{lon}.parquet")
    
    if not force_refresh and os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Error reading cache: {e}")
            
    url = "https://re.jrc.ec.europa.eu/api/tmy"
    params = {
        "lat": lat,
        "lon": lon,
        "outputformat": "json"
    }
    
    try:
        print(f"Fetching TMY data from PVGIS for {lat}, {lon}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        hourly = data['outputs']['tmy_hourly']
        df = pd.DataFrame(hourly)
        
        # Parse time (format: YYYYMMDD:HHmm)
        # TMY years are mixed, so we normalize to a generic year (e.g. 2000) or keep as is and ignore year later.
        # For simplicity, we'll keep the raw string and parse it when needed, or parse to a generic datetime.
        # Actually, let's parse to a datetime object but be aware the year is "typical" (mixed).
        df['time_str'] = df['time(UTC)']
        df['Time'] = pd.to_datetime(df['time(UTC)'], format='%Y%m%d:%H%M')
        
        # Save to cache
        df.to_parquet(cache_file)
        return df
        
    except Exception as e:
        print(f"Error fetching PVGIS data: {e}")
        return pd.DataFrame()

def solar_from_ghi(ghi_series, capacity_mw, efficiency=0.85):
    """
    Estimates solar generation from Global Horizontal Irradiance (GHI).
    
    Args:
        ghi_series (pd.Series): GHI in W/m2.
        capacity_mw (float): Installed capacity in MW.
        efficiency (float): System efficiency (losses, inverter, temp, etc.).
        
    Returns:
        pd.Series: Generation in MW.
    """
    # Standard Test Conditions (STC): 1000 W/m2
    # Generation = Capacity * (GHI / 1000) * Efficiency
    gen_mw = capacity_mw * (ghi_series / 1000.0) * efficiency
    return gen_mw.clip(lower=0.0, upper=capacity_mw)

def wind_from_speed(speed_series, capacity_mw):
    """
    Estimates wind generation from wind speed using a generic power curve.
    
    Args:
        speed_series (pd.Series): Wind speed in m/s.
        capacity_mw (float): Installed capacity in MW.
        
    Returns:
        pd.Series: Generation in MW.
    """
    # Generic 2MW turbine power curve scaled to capacity
    # Cut-in: 3 m/s, Rated: 12 m/s, Cut-out: 25 m/s
    
    def power_curve(v):
        if v < 3.0:
            return 0.0
        elif v < 12.0:
            # Cubic ramp up
            # P ~ v^3
            # Normalized: ((v - 3) / (12 - 3))^3
            return ((v - 3.0) / 9.0) ** 3
        elif v < 25.0:
            return 1.0 # Rated power
        else:
            return 0.0 # Cut-out
            
    # Vectorize
    normalized_power = speed_series.apply(power_curve)
    return normalized_power * capacity_mw

def get_profile_for_year(year, tech, capacity_mw, lat=32.4487, lon=-99.7331):
    """
    Generates a full year profile for the specified year using TMY data.
    
    Args:
        year (int): Target year (e.g., 2024).
        tech (str): "Solar" or "Wind".
        capacity_mw (float): Capacity in MW.
        lat, lon: Location coordinates.
        
    Returns:
        pd.Series: Generation profile aligned with the target year's 15-min intervals (UTC).
    """
    # 1. Get TMY Data
    df_tmy = get_tmy_data(lat, lon)
    if df_tmy.empty:
        return pd.Series()
        
    # 2. Extract relevant column and calculate MW
    if tech == "Solar":
        # Use G(h) - Global irradiance on the horizontal plane
        if 'G(h)' in df_tmy.columns:
            # PVGIS TMY is hourly.
            mw_hourly = solar_from_ghi(df_tmy['G(h)'], capacity_mw)
        else:
            return pd.Series()
    elif tech == "Wind":
        # Use WS10m - Wind speed at 10m
        # Note: Hub height is usually higher (80m+). 
        # Simple shear extrapolation: v_h = v_ref * (h / h_ref)^alpha
        # alpha ~ 0.143 (1/7 power law)
        # h = 80m, h_ref = 10m
        # factor = (80/10)^0.143 = 8^0.143 ≈ 1.35
        if 'WS10m' in df_tmy.columns:
            wind_speed_hub = df_tmy['WS10m'] * 1.35
            mw_hourly = wind_from_speed(wind_speed_hub, capacity_mw)
        else:
            return pd.Series()
    else:
        return pd.Series()
        
    # 3. Align with Target Year
    # TMY data is 8760 hours. We need to map this to the target year's datetime index.
    # We'll create a full year index for the target year (15-min intervals).
    
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31 23:45"
    target_index = pd.date_range(start=start_date, end=end_date, freq='15min', tz='UTC')
    
    # Create a source index from the TMY data, but ignoring the year
    # TMY data has 'Time' column with mixed years.
    # We'll extract month-day-hour-minute and map.
    
    # Simpler approach:
    # 1. Resample TMY (hourly) to 15-min (interpolate).
    # 2. Create a generic "year" index (e.g. 2000, leap year handling might be tricky).
    # 3. Map to target year.
    
    # Let's assume TMY is 8760 points (non-leap).
    # If target is leap, we'll pad.
    
    # Interpolate to 15-min
    # Create a dummy index for interpolation
    dummy_index_hourly = pd.date_range(start="2000-01-01", periods=len(mw_hourly), freq='h')
    s_hourly = pd.Series(mw_hourly.values, index=dummy_index_hourly)
    
    # Resample to 15 min
    s_15min = s_hourly.resample('15min').interpolate(method='linear')
    
    # Now we have ~35040 points.
    # We need to match `target_index` length.
    
    # If target is leap year (366 days), we need more data.
    # If target is non-leap (365 days), we match.
    
    # Simple logic: Tile or truncate/pad.
    # Since TMY is "typical", repeating the last day or Feb 28th for leap day is fine.
    
    values = s_15min.values
    target_len = len(target_index)
    
    if len(values) < target_len:
        # Pad (likely leap year in target)
        # Pad with last value or wrap around
        diff = target_len - len(values)
        values = np.pad(values, (0, diff), mode='edge')
    elif len(values) > target_len:
        # Truncate
        values = values[:target_len]
        
    return pd.Series(values, index=target_index, name="Gen_MW")

if __name__ == "__main__":
    # Test
    print("Testing fetch_tmy...")
    s = get_profile_for_year(2024, "Solar", 100)
    print(f"Solar 2024: {len(s)} points, Max: {s.max():.2f} MW")
    
    w = get_profile_for_year(2024, "Wind", 100)
    print(f"Wind 2024: {len(w)} points, Max: {w.max():.2f} MW")
