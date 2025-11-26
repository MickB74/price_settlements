import requests
import pandas as pd
import json

def get_tmy_data(lat, lon):
    url = "https://re.jrc.ec.europa.eu/api/tmy"
    params = {
        "lat": lat,
        "lon": lon,
        "outputformat": "json"
    }
    try:
        print(f"Fetching TMY data for {lat}, {lon}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Parse JSON
        # Structure: data['outputs']['tmy_hourly'] is a list of dicts
        hourly = data['outputs']['tmy_hourly']
        df = pd.DataFrame(hourly)
        print("Columns:", df.columns)
        print(df.head())
        
        # Check for Solar (G(h)) and Wind (WS10m)
        # G(h): Global irradiance on the horizontal plane (W/m2)
        # WS10m: Wind speed at 10m height (m/s)
        if 'G(h)' in df.columns and 'WS10m' in df.columns:
            print("Success! Found Solar and Wind data.")
            return df
        else:
            print("Missing expected columns.")
            return None
            
    except Exception as e:
        print(f"Error fetching PVGIS data: {e}")
        return None

if __name__ == "__main__":
    # Abilene, Texas
    get_tmy_data(32.4487, -99.7331)
