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

def find_nearest_noaa_station(lat, lon, year=2024):
    """Find nearest NOAA ISD station (Simplified for known Texas hubs or generic search)."""
    # For reliability/speed in this demo, let's hardcode the confirmed stations for our 5 hubs
    # Data from our previous test run
    KNOWN_STATIONS = {
        # North (Waxahachie) -> Mid-Way Regional
        (32.3865, -96.8475): {'usaf': '720299', 'wban': '53966'},
        # South (Zapata) -> Zapata County
        (26.9070, -99.2715): {'usaf': '720584', 'wban': '00181'},
        # West (Roscoe) -> Dyess AFB (Abilene) or nearby? Roscoe is near Sweetwater.
        # Let's use Sweetwater (Avenger Field) if available, or finding nearest dynamically.
        # For this implementation, let's include the dynamic search properly to be robust.
        # But to avoid huge dependency on isd-history.csv every run, strict fallback is good.
    }
    
    # Check if exact match (float comparison might be tricky, so generic search is better)
    pass
    
    # Generic Search
    try:
        stations_url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
        # We use a cache for station list if possible, or just fetch (it's 2MB)
        # To be safe and fast, let's just use the nearest finding logic if not cached
        cache_file = os.path.join(CACHE_DIR, "isd-history.csv")
        
        if os.path.exists(cache_file):
            df_stations = pd.read_csv(cache_file)
        else:
            print("Fetching NOAA station list...")
            df_stations = pd.read_csv(stations_url)
            df_stations.to_csv(cache_file, index=False)
            
        df_stations['END'] = pd.to_datetime(df_stations['END'], format='%Y%m%d', errors='coerce')
        df_active = df_stations[df_stations['END'] >= '2023-01-01'].copy()
        
        df_active['DISTANCE'] = np.sqrt((df_active['LAT'] - lat)**2 + (df_active['LON'] - lon)**2)
        closest = df_active.nsmallest(1, 'DISTANCE').iloc[0]
        
        return {
            'usaf': str(closest['USAF']).zfill(6),
            'wban': str(closest['WBAN']).zfill(5),
            'name': closest['STATION NAME']
        }
    except Exception as e:
        print(f"Error finding NOAA station: {e}")
        return None

def get_noaa_2024_data(lat, lon):
    """Fetch 2024 hourly data from NOAA ISD."""
    cache_file = os.path.join(CACHE_DIR, f"noaa_2024_{lat}_{lon}.parquet")
    
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
        
    station = find_nearest_noaa_station(lat, lon)
    if not station:
        return pd.DataFrame()
        
    usaf, wban = station['usaf'], station['wban']
    url = f"https://www.ncei.noaa.gov/data/global-hourly/access/2024/{usaf}{wban}.csv"
    
    print(f"Fetching NOAA 2024 data for {lat}, {lon} from {station['name']}...")
    
    try:
        df = pd.read_csv(url, low_memory=False)
        df['datetime'] = pd.to_datetime(df['DATE'])
        
        # Parse Wind Speed (WND: angle,quality,type,speed,quality)
        # Speed is scaled by 10 (e.g. 35 = 3.5 m/s). 9999 = Missing.
        def parse_wind(wnd_str):
            try:
                parts = str(wnd_str).split(',')
                if len(parts) >= 4:
                    speed = float(parts[3])
                    if speed == 9999: return np.nan
                    return speed / 10.0
            except:
                pass
            return np.nan
            
        if 'WND' in df.columns:
            df['wind_speed_mps'] = df['WND'].apply(parse_wind)
        else:
            df['wind_speed_mps'] = np.nan
            
        # Select and save
        out_df = df[['datetime', 'wind_speed_mps']].copy()
        out_df = out_df.dropna(subset=['wind_speed_mps'])
        out_df.to_parquet(cache_file)
        return out_df
        
    except Exception as e:
        print(f"Error fetching NOAA data: {e}")
        return pd.DataFrame()

def get_profile_for_year(year, tech, capacity_mw, lat=32.4487, lon=-99.7331):
    """
    Generates a full year profile.
    Uses Actual data for 2005-2023 (PVGIS).
    Uses NOAA ISD for 2024 (Wind only).
    Uses TMY data for future years or fallback.
    """
    # Determine Data Source
    use_pvgis_actual = 2005 <= year <= 2023
    use_noaa_2024 = (year == 2024)
    
    df_data = pd.DataFrame()
    source_type = "TMY" # Default
    
    # 1. Try PVGIS Actuals (2005-2023)
    if use_pvgis_actual:
        # (Existing PVGIS fetch logic...)
        try:
            # Re-implement get_actual_data inline or call it
            # For brevity in this edit, leveraging existing logic structure if feasible
            # But we must ensure df_data is populated
            
            # Using the cache/fetch logic from previous block:
            cache_file = os.path.join(CACHE_DIR, f"actual_{year}_{lat}_{lon}.parquet")
            if os.path.exists(cache_file):
                 df_data = pd.read_parquet(cache_file)
            else:
                # Fetch new
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

    # 2. Try NOAA 2024 (Wind Only)
    if use_noaa_2024 and tech == "Wind" and df_data.empty:
        df_noaa = get_noaa_2024_data(lat, lon)
        if not df_noaa.empty:
            df_data = df_noaa
            # Map columns for downstream compatibility
            # Downstream expects 'time' or 'time(UTC)' and 'WS10m'
            # We have 'datetime' and 'wind_speed_mps'
            df_data['WS10m'] = df_data['wind_speed_mps']
            source_type = "NOAA_Actual"

    # 3. Fallback to TMY (Solar 2024, Future Years, or failures)
    if df_data.empty:
        df_data = get_tmy_data(lat, lon)
        source_type = "TMY"
    
    if df_data.empty:
        return pd.Series()

    # Calculate MW
    if tech == "Solar":
        if source_type == "Actual":
            irradiance = df_data['Gb(i)'] + df_data['Gd(i)'] + df_data.get('Gr(i)', 0)
        elif source_type == "TMY":
            irradiance = df_data['G(h)']
        else:
            return pd.Series() # Should not happen for Solar if logic is correct
            
        mw_hourly = solar_from_ghi(irradiance, capacity_mw)
        
    elif tech == "Wind":
        if 'WS10m' in df_data.columns:
            # Extrapolate to hub height (80m)
            wind_speed_hub = df_data['WS10m'] * 1.35
            mw_hourly = wind_from_speed(wind_speed_hub, capacity_mw)
        else:
            return pd.Series()
    else:
        return pd.Series()

    # Align with Target Year (Resampling)
    if source_type in ["Actual", "NOAA_Actual"]:
        # Actual data is hourly for the specific year. Upsample to 15-min.
        
        # Parse timestamps
        if source_type == "NOAA_Actual":
            # Already datetime objects
            timestamps = df_data['datetime']
            # Ensure UTC timezone if not present (NOAA is UTC usually)
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
        
        # Resample to 15min
        # Handle duplicate indices if any (NOAA sometimes has dups)
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
    print("Testing fetch_tmy (Hybrid + NOAA)...")
    
    # Test Actual (2020)
    print("\n--- Test 2020 Solar (PVGIS Actual) ---")
    s_2020 = get_profile_for_year(2020, "Solar", 100)
    print(f"Solar 2020: {len(s_2020)} points, Max: {s_2020.max():.2f} MW")
    
    # Test NOAA 2024 (Wind)
    print("\n--- Test 2024 Wind (NOAA Actual) ---")
    s_2024 = get_profile_for_year(2024, "Wind", 100)
    print(f"Wind 2024: {len(s_2024)} points, Max: {s_2024.max():.2f} MW")
    if not s_2024.empty:
        print(f"Sample head:\n{s_2024.head()}")

    # Test TMY (2025)
    print("\n--- Test 2025 Wind (TMY) ---")
    s_2025 = get_profile_for_year(2025, "Wind", 100)
    print(f"Wind 2025: {len(s_2025)} points, Max: {s_2025.max():.2f} MW")

