import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

def get_eia_api_key():
    return os.getenv('EIA_API_KEY')

def fetch_from_api(plant_id, year, api_key):
    """Fetch monthly generation for a specific plant from EIA API."""
    base_url = "https://api.eia.gov/v2/electricity/facility-fuel/data/"
    params = {
        'api_key': api_key,
        'frequency': 'monthly',
        'data': ['generation'],
        'facets[plantCode][]': plant_id,
        'start': f'{year}-01',
        'end': f'{year}-12',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc'
    }
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'response' in data and 'data' in data['response']:
                df = pd.DataFrame(data['response']['data'])
                if not df.empty:
                    # Clean up
                    df['month'] = pd.to_datetime(df['period']).dt.month
                    df['Net_Gen_MWh'] = pd.to_numeric(df['generation'], errors='coerce')
                    # Sum across all fuel types/prime movers for this plant (usually just one for wind/solar plants)
                    df_agg = df.groupby('month')['Net_Gen_MWh'].sum().reset_index()
                    return df_agg
    except Exception as e:
        print(f"EIA API Error: {e}")
        
    return pd.DataFrame()

def fetch_from_excel_fallback(plant_id, year, file_path=None):
    """Fetch from local EIA-923 Excel file if available."""
    # Look for file in root matching pattern
    if not file_path:
        # Try generic patterns
        candidates = [
            f"EIA923_Schedules_2_3_4_5_M_12_{year}_Final.xlsx",
            "EIA923_Schedules_2_3_4_5_M_12_2024_Final.xlsx" # Fallback to 2024 if specific year missing? or just fail.
        ]
        for c in candidates:
            if os.path.exists(c):
                file_path = c
                break
    
    if not file_path or not os.path.exists(file_path):
        return pd.DataFrame()
        
    try:
        # Load specific sheet "Page 1 Generation and Fuel Data"
        # Skip header rows (usually first 5 rows are headers)
        df = pd.read_excel(file_path, sheet_name='Page 1 Generation and Fuel Data', skiprows=5)
        
        # Filter by Plant Id
        df_plant = df[df['Plant Id'] == int(plant_id)].copy()
        
        if df_plant.empty:
            return pd.DataFrame()
            
        # Columns correspond to months. Need to identify them.
        # Usually: Net Generation (Megawatthours) -> then columns for Jan, Feb...
        # This is tricky because the layout changes. 
        # API is infinitely safer. This is a "Hail Mary" fallback.
        pass
    except Exception:
        pass
        
    return pd.DataFrame()

def get_plant_generation(plant_id, year):
    """
    Get monthly Net Generation (MWh) for valid months 1-12.
    Returns: pd.Series index=month (1-12), value=MWh
    """
    api_key = get_eia_api_key()
    
    df = pd.DataFrame()
    
    if api_key:
        df = fetch_from_api(plant_id, year, api_key)
        
    if df.empty:
        # Try fallback (implementation pending full excel parsing logic)
        # For now, just return empty if API fails, or load a cached CSV if we built one.
        pass
        
    if df.empty:
        return None
        
    # Ensure full 1-12 index
    s = df.set_index('month')['Net_Gen_MWh']
    s = s.reindex(range(1, 13), fill_value=0)
    return s

def find_plant_id(name):
    """
    Look up EIA Plant ID by name substring.
    Requires a mapping file or a search.
    """
    # Placeholder: Map known ERCOT assets to EIA IDs
    # Ideally we'd load `eia_tx_wind_2023_summary.csv` if it exists
    mapping_file = 'eia_tx_wind_2023_summary.csv'
    if os.path.exists(mapping_file):
        df = pd.read_csv(mapping_file)
        # Search
        # df columns: plantCode, plantName, ...
        matches = df[df['plantName'].str.contains(name, case=False, na=False)]
        if not matches.empty:
            return matches.iloc[0]['plantCode'], matches.iloc[0]['plantName']
            
    return None, None
