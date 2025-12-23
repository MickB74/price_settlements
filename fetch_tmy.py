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


def get_openmeteo_2024_data(lat, lon):
    """Fetch 2024 hourly solar and wind data from Open-Meteo."""
    cache_file = os.path.join(CACHE_DIR, f"openmeteo_2024_{lat}_{lon}.parquet")
    
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
        
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "hourly": "shortwave_radiation,wind_speed_100m",
        "timezone": "UTC"
    }
    
    print(f"Fetching Open-Meteo 2024 data for {lat}, {lon}...")
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Open-Meteo Error: {response.text}")
            return pd.DataFrame()
            
        data = response.json()
        df = pd.DataFrame(data['hourly'])
        df['datetime'] = pd.to_datetime(df['time'])
        
        # Rename columns to standard internal names
        # shortwave_radiation = GHI (W/m2)
        # wind_speed_100m is in km/h -> Convert to m/s
        df['GHI_Wm2'] = df['shortwave_radiation']
        df['Wind_Speed_100m_mps'] = df['wind_speed_100m'] / 3.6
        
        out_df = df[['datetime', 'GHI_Wm2', 'Wind_Speed_100m_mps']].copy()
        out_df.to_parquet(cache_file)
        return out_df
        
    except Exception as e:
        print(f"Error fetching Open-Meteo data: {e}")
        return pd.DataFrame()

def get_profile_for_year(year, tech, capacity_mw, lat=32.4487, lon=-99.7331):
    """
    Generates a full year profile.
    Uses Actual data for 2005-2023 (PVGIS).
    Uses Open-Meteo for 2024 (Solar + Wind).
    Uses TMY data for future years or fallback.
    """
    # Determine Data Source
    use_pvgis_actual = 2005 <= year <= 2023
    use_openmeteo_2024 = (year == 2024)
    
    df_data = pd.DataFrame()
    source_type = "TMY" # Default
    
    # 1. Try PVGIS Actuals (2005-2023)
    if use_pvgis_actual:
        try:
            cache_file = os.path.join(CACHE_DIR, f"actual_{year}_{lat}_{lon}.parquet")
            if os.path.exists(cache_file):
                 df_data = pd.read_parquet(cache_file)
            else:
                url = "https://re.jrc.ec.europa.eu/api/seriescalc"
                params = {
                    "lat": lat, "lon": lon, "startyear": year, "endyear": year,
                    "outputformat": "json", "pvcalculation": 0, "components": 1,
                    "angle": 0 # Horizontal
                }
                print(f"Fetching PVGIS Actual data for {year}...")
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if 'outputs' in data and 'hourly' in data['outputs']:
                        df_data = pd.DataFrame(data['outputs']['hourly'])
                        df_data.to_parquet(cache_file)
                else:
                    print(f"PVGIS Error: {response.status_code}")
            
            if not df_data.empty:
                source_type = "Actual"
        except Exception as e:
            print(f"Error in PVGIS flow: {e}")

    # 2. Try Open-Meteo 2024 (Solar + Wind)
    if use_openmeteo_2024 and df_data.empty:
        df_om = get_openmeteo_2024_data(lat, lon)
        if not df_om.empty:
            df_data = df_om
            source_type = "OpenMeteo_Actual"

    # 3. Fallback to TMY
    if df_data.empty:
        df_data = get_tmy_data(lat, lon)
        source_type = "TMY"
    
    if df_data.empty:
        return pd.Series()

    # Calculate MW from Weather Data
    if tech == "Solar":
        if source_type == "Actual":
            # PVGIS: G(h) = Gb(i) + Gd(i) (approx on horizontal)
            irradiance = df_data['Gb(i)'] + df_data['Gd(i)'] + df_data.get('Gr(i)', 0)
        elif source_type == "OpenMeteo_Actual":
            # Open-Meteo: GHI provided directly
            irradiance = df_data['GHI_Wm2']
        elif source_type == "TMY":
            # PVGIS TMY: G(h)
            irradiance = df_data['G(h)']
        else:
            return pd.Series()
            
        mw_hourly = solar_from_ghi(irradiance, capacity_mw)
        
    elif tech == "Wind":
        if source_type in ["Actual", "TMY"]:
            if 'WS10m' in df_data.columns:
                # PVGIS TMY/Actual: 10m wind speed
                # Extrapolate to hub height (80m) using power law (alpha=0.143 standard, or empirical 1.35 factor)
                wind_speed_hub = df_data['WS10m'] * 1.35
                mw_hourly = wind_from_speed(wind_speed_hub, capacity_mw)
            else:
                return pd.Series()
        elif source_type == "OpenMeteo_Actual":
            # Open-Meteo: 100m wind speed provided directly
            # This is much better than extrapolating from 10m!
            # We'll treat 100m speed as roughly equivalent to 80-100m hub height
            mw_hourly = wind_from_speed(df_data['Wind_Speed_100m_mps'], capacity_mw)
        else:
            return pd.Series()
    else:
        return pd.Series()

    # Align with Target Year (Resampling)
    if source_type in ["Actual", "OpenMeteo_Actual"]:
        # Parse timestamps
        if source_type == "OpenMeteo_Actual":
            timestamps = df_data['datetime']
            if timestamps.dt.tz is None:
                timestamps = timestamps.dt.tz_localize('UTC')
            else:
                timestamps = timestamps.dt.tz_convert('UTC')
        else:
            # PVGIS format
            if 'time' in df_data.columns: time_col = 'time'
            elif 'time(UTC)' in df_data.columns: time_col = 'time(UTC)'
            else: return pd.Series()
            timestamps = pd.to_datetime(df_data[time_col], format='%Y%m%d:%H%M', utc=True)
        
        # Create Series
        s_hourly = pd.Series(mw_hourly.values, index=timestamps)
        
        # Handle duplicates/Resample
        s_hourly = s_hourly[~s_hourly.index.duplicated(keep='first')]
        s_15min = s_hourly.resample('15min').interpolate(method='linear')
        
        # Reindex to full year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31 23:45"
        target_index = pd.date_range(start=start_date, end=end_date, freq='15min', tz='UTC')
        
        s_final = s_15min.reindex(target_index).ffill().bfill()
        return s_final.fillna(0)

        
    else:
        # TMY Logic (Linear Interpolation of typical year to target year)
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
    print("Testing fetch_tmy (Hybrid + Open-Meteo)...")
    
    # Test Actual (2020)
    print("\n--- Test 2020 Solar (PVGIS Actual) ---")
    s_2020 = get_profile_for_year(2020, "Solar", 100)
    print(f"Solar 2020: {len(s_2020)} points, Max: {s_2020.max():.2f} MW")
    
    # Test Open-Meteo 2024 (Wind)
    print("\n--- Test 2024 Wind (Open-Meteo 100m) ---")
    s_2024_wind = get_profile_for_year(2024, "Wind", 100)
    print(f"Wind 2024: {len(s_2024_wind)} points, Max: {s_2024_wind.max():.2f} MW")
    if not s_2024_wind.empty:
        print(f"Sample head:\n{s_2024_wind.head()}")

    # Test Open-Meteo 2024 (Solar)
    print("\n--- Test 2024 Solar (Open-Meteo GHI) ---")
    s_2024_solar = get_profile_for_year(2024, "Solar", 100)
    print(f"Solar 2024: {len(s_2024_solar)} points, Max: {s_2024_solar.max():.2f} MW")
    if not s_2024_solar.empty:
        print(f"Sample head:\n{s_2024_solar.head()}")

    # Test TMY (2025)
    print("\n--- Test 2025 Wind (TMY) ---")
    s_2025 = get_profile_for_year(2025, "Wind", 100)
    print(f"Wind 2025: {len(s_2025)} points, Max: {s_2025.max():.2f} MW")


