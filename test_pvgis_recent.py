import requests
import pandas as pd

def test_specific_year(year, lat=32.4487, lon=-99.7331):
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
        print(f"Fetching {year} data from PVGIS...")
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return
            
        data = response.json()
        if 'outputs' in data and 'hourly' in data['outputs']:
            print(f"Success! Fetched {len(data['outputs']['hourly'])} rows for {year}.")
        else:
            print(f"Unexpected response for {year}.")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_specific_year(2024)
    test_specific_year(2025)
