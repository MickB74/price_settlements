import pandas as pd
import numpy as np
import streamlit as st
import sys
import os

# Ensure parent directory is in path to import fetch_tmy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch_tmy

def run_historical_analysis(lat, lon, tech="Solar", capacity_mw=100.0, losses_pct=14.0, turbine_type="GENERIC", progress_bar=None):
    """
    Runs generation model for years 2005-2024 to capture weather variability.
    Returns:
        - results_df: DataFrame with Year, Annual_MWh, Capacity_Factor
        - stats: Dictionary with P-values (P99, P90, P50, P10, P1)
    """
    
    # Range of years to model
    # PVGIS Actuals: 2005-2023
    # Open-Meteo: 2024
    years = range(2005, 2025) 
    results = []
    
    efficiency = 1.0 # Request Gross from backend
    # We apply losses later or pass them in? 
    # fetch_tmy now takes efficiency. If we pass it there, we get net.
    # The App applies losses *after*.
    # Let's align with App: Get Gross from backend (efficiency=1.0), then apply losses here.
    
    loss_factor = (1 - losses_pct / 100.0)

    total_steps = len(years)
    
    for i, year in enumerate(years):
        if progress_bar:
            progress_bar.progress((i + 1) / total_steps, text=f"Simulating {year}...")
            
        try:
            # Fetch Profile (Gross)
            profile = fetch_tmy.get_profile_for_year(
                year=year,
                tech=tech,
                capacity_mw=capacity_mw,
                lat=lat,
                lon=lon,
                turbine_type=turbine_type,
                efficiency=1.0,
                force_tmy=False # Ensure we get ACTUALS
            )
            
            if not profile.empty:
                # Calculate Annual Sum
                annual_gross_mwh = profile.sum() / 4.0 # Assuming 15-min data (sum MW / 4)
                # double check freq? fetch_tmy returns 15-min resampled.
                
                # Apply Losses
                annual_net_mwh = annual_gross_mwh * loss_factor
                
                results.append({
                    'Year': year,
                    'Annual_MWh': annual_net_mwh,
                    'Capacity_Factor': (annual_net_mwh) / (capacity_mw * 8760) * 100
                })
                
        except Exception as e:
            print(f"Error simulating {year}: {e}")
            continue

    if not results:
        return pd.DataFrame(), {}
        
    df = pd.DataFrame(results)
    
    # Calculate Probabilities (P-values)
    # P90: Exceeded 90% of the time (Conservative) -> 10th percentile of distribution?
    # Actually standard definition: P90 = Value where 90% of probabilities are higher.
    # So P90 is the 10th percentile of the values (Low case).
    # P10 is the 90th percentile (High upside case).
    
    values = df['Annual_MWh']
    stats = {
        'P99': np.percentile(values, 1),
        'P90': np.percentile(values, 10),
        'P75': np.percentile(values, 25),
        'P50': np.percentile(values, 50), # Median
        'P25': np.percentile(values, 75),
        'P10': np.percentile(values, 90),
        'P1':  np.percentile(values, 99),
        'Mean': np.mean(values),
        'StdDev': np.std(values),
        'Min': np.min(values),
        'Max': np.max(values)
    }
    
    return df, stats
