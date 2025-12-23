"""
Fetch 2024 weather data from NOAA ISD for VPPA generation profile creation.

NOAA Integrated Surface Database (ISD) provides hourly meteorological observations
including wind speed, temperature, and solar radiation proxy data.

API Documentation: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
Dataset: https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database

Usage:
    python fetch_noaa_2024.py
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Texas hub locations
HUB_LOCATIONS = {
    "North": (32.3865, -96.8475),   # Waxahachie
    "South": (26.9070, -99.2715),   # Zapata
    "West": (32.4518, -100.5371),   # Roscoe
    "Houston": (29.3013, -94.7977), # Galveston
    "Panhandle": (35.2220, -101.8313), # Amarillo
}

def find_nearest_station(lat, lon, year=2024):
    """
    Find nearest NOAA ISD weather station to given coordinates.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    year : int
        Year to check for data availability
    
    Returns:
    --------
    dict : Station info with ID and distance
    """
    # NOAA ISD station inventory
    stations_url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
    
    print(f"Finding nearest station to ({lat}, {lon})...")
    
    try:
        df_stations = pd.read_csv(stations_url)
        
        # Filter to stations with recent data
        df_stations['END'] = pd.to_datetime(df_stations['END'], format='%Y%m%d', errors='coerce')
        # Active stations (data through at least 2023)
        df_active = df_stations[df_stations['END'] >= '2023-01-01'].copy()
        
        # Calculate distance (simple Euclidean for nearby stations)
        df_active['DISTANCE'] = np.sqrt(
            (df_active['LAT'] - lat)**2 + 
            (df_active['LON'] - lon)**2
        )
        
        # Get nearest 5 stations
        nearest = df_active.nsmallest(5, 'DISTANCE')
        
        print("\nNearest stations:")
        for idx, row in nearest.iterrows():
            print(f"  {row['STATION NAME']:40s} - {row['DISTANCE']:.3f}° away - {row['USAF']}-{row['WBAN']}")
        
        # Return closest
        closest = nearest.iloc[0]
        return {
            'usaf': str(closest['USAF']).zfill(6),
            'wban': str(closest['WBAN']).zfill(5),
            'name': closest['STATION NAME'],
            'distance': closest['DISTANCE'],
            'lat': closest['LAT'],
            'lon': closest['LON']
        }
        
    except Exception as e:
        print(f"Error finding station: {e}")
        return None

def fetch_noaa_hourly_data(station_id, year=2024):
    """
    Fetch hourly weather data from NOAA ISD.
    
    Parameters:
    -----------
    station_id : dict
        Station info from find_nearest_station()
    year : int
        Year to fetch
    
    Returns:
    --------
    pandas.DataFrame : Hourly weather data
    """
    usaf = station_id['usaf']
    wban = station_id['wban']
    
    # NOAA ISD data URL pattern
    url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{usaf}{wban}.csv"
    
    print(f"\nFetching data from: {url}")
    
    try:
        df = pd.read_csv(url, low_memory=False)
        print(f"✅ Downloaded {len(df)} hourly records for {year}")
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['DATE'])
        
        # Extract wind speed (WND format: "angle,quality,type,speed,quality")
        # Speed is in m/s * 10, and 9999 = missing
        if 'WND' in df.columns:
            def parse_wind(wnd_str):
                try:
                    parts = str(wnd_str).split(',')
                    if len(parts) >= 4:
                        speed = float(parts[3])
                        # 9999 = missing data
                        if speed == 9999:
                            return np.nan
                        # Scale by 10 (e.g., 35 = 3.5 m/s)
                        return speed / 10.0
                except:
                    return np.nan
                return np.nan
            
            df['wind_speed_mps'] = df['WND'].apply(parse_wind)
        else:
            df['wind_speed_mps'] = np.nan
        
        # Extract temperature (TMP format: "temp,quality")  
        # Temp is in C * 10, and +9999 = missing
        if 'TMP' in df.columns:
            def parse_temp(tmp_str):
                try:
                    parts = str(tmp_str).split(',')
                    if len(parts) >= 1:
                        temp = float(parts[0])
                        # +9999 or -9999 = missing data
                        if abs(temp) >= 9999:
                            return np.nan
                        # Scale by 10 (e.g., 195 = 19.5°C)
                        return temp / 10.0
                except:
                    return np.nan
                return np.nan
            
            df['temp_c'] = df['TMP'].apply(parse_temp)
        else:
            df['temp_c'] = np.nan
        
        # Clean data
        result = df[['datetime', 'wind_speed_mps', 'temp_c']].copy()
        
        # Report data quality
        wind_valid = result['wind_speed_mps'].notna().sum()
        temp_valid = result['temp_c'].notna().sum()
        print(f"   Valid wind: {wind_valid}/{len(result)} ({wind_valid/len(result)*100:.1f}%)")
        print(f"   Valid temp: {temp_valid}/{len(result)} ({temp_valid/len(result)*100:.1f}%)")
        
        return result
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def test_2024_availability():
    """Test if 2024 data is available for Texas hub locations."""
    
    print("=" * 70)
    print("TESTING NOAA ISD 2024 DATA AVAILABILITY FOR TEXAS HUBS")
    print("=" * 70)
    
    for hub_name, (lat, lon) in HUB_LOCATIONS.items():
        print(f"\n{'=' * 70}")
        print(f"HUB: {hub_name} ({lat}, {lon})")
        print('=' * 70)
        
        # Find nearest station
        station = find_nearest_station(lat, lon)
        
        if station:
            # Try to fetch 2024 data
            df = fetch_noaa_hourly_data(station, year=2024)
            
            if df is not None and len(df) > 0:
                print(f"\n✅ {hub_name} Hub: 2024 data AVAILABLE!")
                print(f"   Records: {len(df)}")
                print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                print(f"   Avg wind speed: {df['wind_speed_mps'].mean():.2f} m/s")
                print(f"   Avg temp: {df['temp_c'].mean():.2f}°C")
            else:
                print(f"\n⚠️ {hub_name} Hub: No 2024 data yet")
        
        print()

if __name__ == "__main__":
    test_2024_availability()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("If 2024 data is available, we can:")
    print("1. Integrate this into fetch_tmy.py as a fallback")
    print("2. Use actual 2024 weather instead of TMY synthetic")
    print("3. Provide more accurate 2024 generation profiles")
    print("\nNote: NOAA ISD data quality varies by station.")
    print("Some stations may have gaps or missing parameters.")
