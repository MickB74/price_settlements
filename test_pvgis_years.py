import requests
import pandas as pd

def test_specific_year(year, lat=32.4487, lon=-99.7331):
    # PVGIS Hourly Data Endpoint
    url = "https://re.jrc.ec.europa.eu/api/seriescalc"
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": year,
        "endyear": year,
        "outputformat": "json",
        "pvcalculation": 0, # Just radiation/weather
        "components": 1 # Include components like G(h)
    }
    
    try:
        print(f"Fetching {year} data from PVGIS...")
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return
            
        data = response.json()
        
        # Check outputs
        if 'outputs' in data and 'hourly' in data['outputs']:
            hourly = data['outputs']['hourly']
            df = pd.DataFrame(hourly)
            print(f"Success! Fetched {len(df)} rows for {year}.")
            print("Columns:", df.columns)
            print(df.head())
            
            # Check for G(h) and WS10m
            if 'G(h)' in df.columns and 'WS10m' in df.columns:
                print("Found Solar (G(h)) and Wind (WS10m) data.")
            else:
                print("Missing expected columns.")
        else:
            print("Unexpected response structure.")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_specific_year(2020)
    test_specific_year(2023)
