"""
Fetch 2024 weather data (Solar & Wind) from Open-Meteo API.

Open-Meteo provides historical weather data based on reanalysis models (ERA5) and archive models.
It includes Solar Irradiance (GHI), which NOAA ISD lacks.

Docs: https://open-meteo.com/en/docs/historical-weather-api

Usage:
    python fetch_openmeteo_2024.py
"""

import requests
import pandas as pd
import time

# Texas hub locations
HUB_LOCATIONS = {
    "North": (32.3865, -96.8475),   # Waxahachie
    "South": (26.9070, -99.2715),   # Zapata
    "West": (32.4518, -100.5371),   # Roscoe
    "Houston": (29.3013, -94.7977), # Galveston
    "Panhandle": (35.2220, -101.8313), # Amarillo
}

def fetch_openmeteo_data(lat, lon, year=2024):
    """
    Fetch hourly solar and wind data for 2024.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": "temperature_2m,wind_speed_10m,wind_speed_100m,shortwave_radiation,direct_radiation,diffuse_radiation",
        "timezone": "UTC"
    }
    
    print(f"Fetching Open-Meteo data for {lat}, {lon}...")
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None
            
        data = response.json()
        
        # Parse hourly data
        hourly = data['hourly']
        df = pd.DataFrame(hourly)
        
        # Convert time
        df['datetime'] = pd.to_datetime(df['time'])
        
        # Rename columns for clarity
        # shortwave_radiation = GHI (Global Horizontal Irradiance)
        df_clean = df.rename(columns={
            'wind_speed_10m': 'Wind_Speed_10m_mps',
            'wind_speed_100m': 'Wind_Speed_100m_mps', # Useful for hub height!
            'shortwave_radiation': 'GHI_Wm2',
            'direct_radiation': 'DNI_Wm2',
            'diffuse_radiation': 'DHI_Wm2',
            'temperature_2m': 'Temp_C'
        })
        
        return df_clean[['datetime', 'GHI_Wm2', 'Wind_Speed_10m_mps', 'Wind_Speed_100m_mps', 'Temp_C']]

    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def test_2024_availability():
    print("=" * 70)
    print("TESTING OPEN-METEO 2024 DATA (SOLAR + WIND)")
    print("=" * 70)
    
    for hub_name, (lat, lon) in HUB_LOCATIONS.items():
        print(f"\nProcessing {hub_name} Hub...")
        df = fetch_openmeteo_data(lat, lon, 2024)
        
        if df is not None:
            print(f"✅ Data fetched: {len(df)} records")
            print(f"   Solar (GHI) Avg: {df['GHI_Wm2'].mean():.2f} W/m2")
            print(f"   Solar Max: {df['GHI_Wm2'].max():.2f} W/m2")
            print(f"   Wind (100m) Avg: {df['Wind_Speed_100m_mps'].mean():.2f} m/s") # 100m is closer to 80m hub height
            print(f"   Date Range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        time.sleep(1) # Be nice to API

if __name__ == "__main__":
    test_2024_availability()
