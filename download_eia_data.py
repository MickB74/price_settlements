"""
Download EIA-923 plant-level generation data using EIA API v2.

Setup:
1. Get free API key from: https://www.eia.gov/opendata/register.php
2. Set environment variable: export EIA_API_KEY="your_key_here"
3. Run this script

API Documentation: https://www.eia.gov/opendata/
"""

import os
import requests
import pandas as pd
from datetime import datetime

def get_eia_plant_data(api_key, state='TX', year=2023, fuel_type='WND'):
    """
    Fetch plant-level generation data from EIA API v2.
    
    Parameters:
    -----------
    api_key : str
        Your EIA API key
    state : str
        State code (e.g., 'TX' for Texas)
    year : int
        Year to fetch
    fuel_type : str
        'WND' for wind, 'SUN' for solar, 'ALL' for all types
    
    Returns:
    --------
    pandas.DataFrame
        Monthly generation data by plant
    """
    
    base_url = "https://api.eia.gov/v2/electricity/facility-fuel/data/"
    
    params = {
        'api_key': api_key,
        'frequency': 'monthly',
        'data': ['generation'],  # MWh
        'facets[state][]': state,
        'start': f'{year}-01',
        'end': f'{year}-12',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'offset': 0,
        'length': 5000,  # Max per request
    }
    
    if fuel_type != 'ALL':
        params['facets[fueltypeid][]'] = fuel_type
    
    print(f"Fetching EIA data: {state} {fuel_type} {year}...")
    
    all_data = []
    offset = 0
    
    while True:
        params['offset'] = offset
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            break
        
        data = response.json()
        
        if 'response' not in data or 'data' not in data['response']:
            print("No data returned")
            break
        
        records = data['response']['data']
        if not records:
            break
        
        all_data.extend(records)
        print(f"  Fetched {len(all_data)} records...")
        
        # Check if there's more data
        total = data['response'].get('total', 0)
        if len(all_data) >= total:
            break
        
        offset += len(records)
    
    if not all_data:
        print("No data found!")
        return None
    
    df = pd.DataFrame(all_data)
    print(f"✅ Downloaded {len(df)} records")
    
    return df

def summarize_by_plant(df):
    """Pivot data to show monthly generation by plant."""
    # Convert period to month number
    df['month'] = pd.to_datetime(df['period']).dt.month
    df['generation_mwh'] = pd.to_numeric(df['generation'], errors='coerce')
    
    # Pivot: plants as rows, months as columns
    pivot = df.pivot_table(
        values='generation_mwh',
        index=['plantCode', 'plantName', 'state'],
        columns='month',
        aggfunc='sum'
    )
    
    pivot['Total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('Total', ascending=False)
    
    return pivot

if __name__ == "__main__":
    # Get API key from environment
    api_key = os.getenv('EIA_API_KEY')
    
    if not api_key:
        print("❌ EIA API key not found!")
        print("\nTo use this script:")
        print("1. Get API key from: https://www.eia.gov/opendata/register.php")
        print("2. Run: export EIA_API_KEY='your_key_here'")
        print("3. Run this script again")
        exit(1)
    
    # Fetch Texas wind data for 2023
    df = get_eia_plant_data(
        api_key=api_key,
        state='TX',
        year=2023,
        fuel_type='WND'  # Wind
    )
    
    if df is not None:
        # Save raw data
        df.to_csv('eia_tx_wind_2023.csv', index=False)
        print(f"\nSaved to: eia_tx_wind_2023.csv")
        
        # Summarize by plant
        summary = summarize_by_plant(df)
        summary.to_csv('eia_tx_wind_2023_summary.csv')
        
        print("\n=== TOP 10 WIND PLANTS (2023 Annual MWh) ===")
        print(summary.head(10)[['Total']])
        
        print(f"\nTotal records: {len(df)}")
        print(f"Unique plants: {df['plantCode'].nunique()}")
