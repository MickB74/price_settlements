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
        
        # Standardize columns
        # TMY has 'G(h)', 'WS10m'
        # We'll keep them as is for now, but ensure consistency with actuals
        
        df['time_str'] = df['time(UTC)']
        df['Time'] = pd.to_datetime(df['time(UTC)'], format='%Y%m%d:%H%M')
        
        # Save to cache
        df.to_parquet(cache_file)
        return df
        
    except Exception as e:
        print(f"Error fetching PVGIS TMY data: {e}")
        return pd.DataFrame()

def get_actual_data(year, lat=32.4487, lon=-99.7331, force_refresh=False):
    """
    Fetches actual hourly data for a specific year from PVGIS.
    """
    cache_file = os.path.join(CACHE_DIR, f"actual_{year}_{lat}_{lon}.parquet")
    
    if not force_refresh and os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Error reading cache: {e}")
            
    url = "https://re.jrc.ec.europa.eu/api/seriescalc"
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": year,
        "endyear": year,
        "outputformat": "json",
        "pvcalculation": 0,
        "components": 1
    }
    
    try:
        print(f"Fetching Actual data for {year} from PVGIS...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'outputs' in data and 'hourly' in data['outputs']:
            hourly = data['outputs']['hourly']
            df = pd.DataFrame(hourly)
            
            # Map columns to match TMY if needed, or handle in calculation
            # Actuals: 'Gb(i)', 'Gd(i)', 'Gr(i)', 'H_sun', 'T2m', 'WS10m', 'Int'
            # TMY: 'G(h)', 'Gb(n)', 'Gd(h)', 'IR(h)', 'WS10m', 'WD10m', 'SP'
            
            # We need Global Horizontal Irradiance.
            # seriescalc 'components=1' gives:
            # Gb(i): Beam irradiance on inclined plane (here 0 deg?) -> No, default is optimized slope?
            # Wait, we didn't specify slope/azimuth. Defaults might apply.
            # To get G(h) (Global Horizontal), we should check if it's returned.
            # The previous test showed: 'Gb(i)', 'Gd(i)', 'Gr(i)', 'H_sun', 'T2m', 'WS10m', 'Int'
            # If slope is not 0, Gb(i) is on inclined.
            # Let's try to force horizontal plane to get G(h) equivalent.
            # Or assume G(h) = Gb(i) + Gd(i) + Gr(i) if slope is 0?
            # Actually, let's request "mountingplace=free" and "angle=0" (horizontal).
            
            # Re-fetch with horizontal parameters if needed.
            # But let's check if we can just use what we have.
            # Ideally we want G(h).
            # Let's add 'angle=0' to params to ensure horizontal.
            
            # Save to cache
            df.to_parquet(cache_file)
            return df
        else:
            print(f"No hourly data found for {year}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching PVGIS Actual data: {e}")
        return pd.DataFrame()

def solar_from_ghi(ghi_series, capacity_mw, efficiency=0.85):
    """Estimates solar generation from GHI."""
    gen_mw = capacity_mw * (ghi_series / 1000.0) * efficiency
    return gen_mw.clip(lower=0.0, upper=capacity_mw)

def wind_from_speed(speed_series, capacity_mw):
    """Estimates wind generation from wind speed."""
    def power_curve(v):
        if v < 3.0: return 0.0
        elif v < 12.0: return ((v - 3.0) / 9.0) ** 3
        elif v < 25.0: return 1.0
        else: return 0.0
    
    normalized_power = speed_series.apply(power_curve)
    return normalized_power * capacity_mw

def get_profile_for_year(year, tech, capacity_mw, lat=32.4487, lon=-99.7331):
    """
    Generates a full year profile.
    Uses Actual data for 2005-2023.
    Uses TMY data for other years (fallback).
    """
    # Determine Data Source
    use_actual = 2005 <= year <= 2023
    
    if use_actual:
        # Fetch Actual
        # We need to ensure we get horizontal irradiance for Solar.
        # We'll update get_actual_data to request horizontal.
        # For now, let's call it and handle columns.
        
        # To get G(h) equivalent from seriescalc with angle=0:
        # G(h) = Gb(i) + Gd(i) (since Gr(i) is 0 on horizontal usually)
        # We need to pass angle=0 to get_actual_data.
        # Let's modify get_actual_data to take params or just hardcode horizontal for now.
        
        # Actually, let's modify get_actual_data in place (below).
        pass
    
    # ... (Refactoring to support both in one flow)
    
    df_data = pd.DataFrame()
    source_type = "TMY"
    
    if use_actual:
        # We need to re-implement get_actual_data with angle=0 to be safe
        url = "https://re.jrc.ec.europa.eu/api/seriescalc"
        params = {
            "lat": lat,
            "lon": lon,
            "startyear": year,
            "endyear": year,
            "outputformat": "json",
            "pvcalculation": 0,
            "components": 1,
            "angle": 0, # Horizontal
            "aspect": 0
        }
        
        cache_file = os.path.join(CACHE_DIR, f"actual_{year}_{lat}_{lon}.parquet")
        if os.path.exists(cache_file):
             df_data = pd.read_parquet(cache_file)
        else:
            try:
                print(f"Fetching Actual data for {year}...")
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                if 'outputs' in data and 'hourly' in data['outputs']:
                    df_data = pd.DataFrame(data['outputs']['hourly'])
                    df_data.to_parquet(cache_file)
            except Exception as e:
                print(f"Error fetching actuals: {e}")
        
        source_type = "Actual"

    # Fallback to TMY if actual failed or not requested
    if df_data.empty:
        df_data = get_tmy_data(lat, lon)
        source_type = "TMY"
    
    if df_data.empty:
        return pd.Series()

    # Calculate MW
    if tech == "Solar":
        # Solar Irradiance
        if source_type == "Actual":
            # With angle=0, Gb(i) + Gd(i) should be G(h)
            # Or just sum them.
            # Columns: 'Gb(i)', 'Gd(i)', 'Gr(i)'
            irradiance = df_data['Gb(i)'] + df_data['Gd(i)'] + df_data.get('Gr(i)', 0)
        else:
            # TMY
            irradiance = df_data['G(h)']
            
        mw_hourly = solar_from_ghi(irradiance, capacity_mw)
        
    elif tech == "Wind":
        # Wind Speed
        if 'WS10m' in df_data.columns:
            # Extrapolate to hub height (80m)
            wind_speed_hub = df_data['WS10m'] * 1.35
            mw_hourly = wind_from_speed(wind_speed_hub, capacity_mw)
        else:
            return pd.Series()
    else:
        return pd.Series()

    # Align with Target Year
    # If Actual: The data is already for the correct year (hourly).
    # If TMY: The data is "typical" (mixed years).
    
    if source_type == "Actual":
        # Actual data is hourly for the specific year.
        # We just need to upsample to 15-min.
        
        # Create index from 'time' string?
        # Format: YYYYMMDD:HHMM
        # 20200101:0030 (UTC)
        
        # Parse time
        # Note: PVGIS time is usually UTC.
        if 'time' in df_data.columns:
            time_col = 'time'
        elif 'time(UTC)' in df_data.columns:
            time_col = 'time(UTC)'
        else:
            return pd.Series()
            
        # Parse
        # For Actuals, format is YYYYMMDD:HHMM
        timestamps = pd.to_datetime(df_data[time_col], format='%Y%m%d:%H%M', utc=True)
        
        # Create Series
        s_hourly = pd.Series(mw_hourly.values, index=timestamps)
        
        # Resample to 15min
        s_15min = s_hourly.resample('15min').interpolate(method='linear')
        
        # Ensure it covers the full year (start/end might be slightly off due to center-of-interval timestamps)
        # PVGIS hourly is usually center or end? 00:30 usually means 00:00-01:00 avg?
        # We'll reindex to the exact expected index for the year.
        
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31 23:45"
        target_index = pd.date_range(start=start_date, end=end_date, freq='15min', tz='UTC')
        
        s_final = s_15min.reindex(target_index).fillna(method='ffill').fillna(method='bfill') # Fill edges
        return s_final.fillna(0) # Safety
        
    else:
        # TMY Logic (Same as before)
        # Interpolate to 15-min
        dummy_index_hourly = pd.date_range(start="2000-01-01", periods=len(mw_hourly), freq='h')
        s_hourly = pd.Series(mw_hourly.values, index=dummy_index_hourly)
        s_15min = s_hourly.resample('15min').interpolate(method='linear')
        
        values = s_15min.values
        
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31 23:45"
        target_index = pd.date_range(start=start_date, end=end_date, freq='15min', tz='UTC')
        
        target_len = len(target_index)
        if len(values) < target_len:
            diff = target_len - len(values)
            values = np.pad(values, (0, diff), mode='edge')
        elif len(values) > target_len:
            values = values[:target_len]
            
        return pd.Series(values, index=target_index, name="Gen_MW")

if __name__ == "__main__":
    print("Testing fetch_tmy (Hybrid)...")
    # Test Actual (2020)
    s_2020 = get_profile_for_year(2020, "Solar", 100)
    print(f"Solar 2020 (Actual): {len(s_2020)} points, Max: {s_2020.max():.2f} MW")
    
    # Test TMY (2025)
    s_2025 = get_profile_for_year(2025, "Wind", 100)
    print(f"Wind 2025 (TMY): {len(s_2025)} points, Max: {s_2025.max():.2f} MW")

