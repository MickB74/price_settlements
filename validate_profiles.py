"""
Validate synthetic wind/solar profiles against actual EIA-923 data.

Usage:
1. Download EIA-923 data from https://www.eia.gov/electricity/data/eia923/
2. Extract the "Page 1 Generation and Fuel Data" sheet
3. Filter to Texas wind/solar plants
4. Run this script to compare monthly patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import fetch_tmy

# Our hub locations
HUB_LOCATIONS = {
    "North": (32.3865, -96.8475),   # Waxahachie
    "South": (26.9070, -99.2715),   # Zapata
    "West": (32.4518, -100.5371),   # Roscoe
    "Houston": (29.3013, -94.7977), # Galveston
    "Panhandle": (35.2220, -101.8313), # Amarillo
}

def generate_monthly_synthetic(year, tech, capacity_mw, hub):
    """Generate monthly totals from our synthetic profiles."""
    lat, lon = HUB_LOCATIONS[hub]
    
    # Get full year profile
    profile = fetch_tmy.get_profile_for_year(
        year=year,
        tech=tech,
        capacity_mw=capacity_mw,
        lat=lat,
        lon=lon
    )
    
    # Convert to dataframe
    df = pd.DataFrame({'Gen_MW': profile})
    df['Month'] = df.index.month
    
    # Calculate monthly energy (MWh) - assuming 15-min intervals
    df['Energy_MWh'] = df['Gen_MW'] * 0.25
    
    # Group by month
    monthly = df.groupby('Month')['Energy_MWh'].sum()
    
    return monthly

def compare_to_eia(eia_file_path, plant_name, year=2023, tech='Wind', capacity_mw=100, hub='West'):
    """
    Compare synthetic profile to actual EIA data.
    
    Parameters:
    -----------
    eia_file_path : str
        Path to EIA-923 Excel file or CSV
    plant_name : str
        Name of the plant to analyze (must match EIA data)
    year : int
        Year to analyze
    tech : str
        'Wind' or 'Solar'
    capacity_mw : float
        Plant capacity in MW
    hub : str
        Which hub to use for synthetic generation
    """
    
    print(f"=== VALIDATION: {plant_name} ({year}) ===\n")
    
    # Load EIA data (user will need to clean this)
    print("Loading EIA data...")
    # This is a placeholder - actual implementation depends on EIA file format
    # eia_data = pd.read_excel(eia_file_path, sheet_name='Sheet1')
    # actual_monthly = eia_data[eia_data['Plant Name'] == plant_name].groupby('Month')['Net Generation (MWh)'].sum()
    
    print("EIA data should have columns: Plant Name, Month, Net Generation (MWh)")
    print("You'll need to manually filter and format the EIA data.")
    print("\nFor now, generating synthetic data only...\n")
    
    # Generate synthetic
    print(f"Generating synthetic {tech} profile for {hub} Hub...")
    synthetic_monthly = generate_monthly_synthetic(year, tech, capacity_mw, hub)
    
    print("\n=== SYNTHETIC MONTHLY GENERATION (MWh) ===")
    for month in range(1, 13):
        month_name = datetime(2000, month, 1).strftime('%B')
        print(f"{month_name:>10}: {synthetic_monthly[month]:>10,.0f} MWh")
    
    print(f"\nTotal Year: {synthetic_monthly.sum():>10,.0f} MWh")
    print(f"Capacity Factor: {(synthetic_monthly.sum() / (capacity_mw * 8760) * 100):.1f}%")
    
    # TODO: Add actual comparison when EIA data is loaded
    # print("\n=== COMPARISON ===")
    # print(f"Actual vs Synthetic:")
    # for month in range(1, 13):
    #     actual = actual_monthly[month]
    #     synthetic = synthetic_monthly[month]
    #     diff_pct = ((synthetic - actual) / actual * 100)
    #     print(f"Month {month}: {diff_pct:+.1f}%")
    
    return synthetic_monthly

if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("SYNTHETIC PROFILE VALIDATION")
    print("=" * 60)
    
    # Test West Hub wind
    monthly = compare_to_eia(
        eia_file_path="path/to/eia923.xlsx",  # Update this
        plant_name="Example Wind Farm",        # Update this
        year=2023,
        tech='Wind',
        capacity_mw=100,
        hub='West'
    )
    
    print("\n\nTo use with actual EIA data:")
    print("1. Download EIA-923 from https://www.eia.gov/electricity/data/eia923/")
    print("2. Open 'Page 1 Generation and Fuel Data' sheet")
    print("3. Filter to Texas plants")
    print("4. Export monthly generation data")
    print("5. Update the compare_to_eia() function to load your data")
