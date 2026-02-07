"""
Monte Carlo Simulation Module for VPPA Analysis

Uses historical bootstrap method to generate probabilistic settlement outcomes.
Randomly samples from available weather years (2005-2024) and price years (2020-2026)
to create distribution of possible results.
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple
import sys
import os

# Import required modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch_tmy


def run_bootstrap_simulation(
    scenario_config: Dict,
    n_iterations: int = 1000,
    weather_years: List[int] = None,
    price_years: List[int] = None,
    price_data_cache: Dict[int, pd.DataFrame] = None,
    generation_profile_cache: Dict[int, pd.Series] = None,
    progress_callback=None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run historical bootstrap Monte Carlo simulation for a VPPA scenario.
    
    Args:
        scenario_config: Dict containing:
            - hub: Hub location (e.g., 'HB_NORTH')
            - tech: Technology ('Solar' or 'Wind')
            - capacity_mw: Capacity in MW
            - lat: Latitude
            - lon: Longitude
            - vppa_price: Strike price ($/MWh)
            - revenue_share: Buyer's upside share (0-100%)
            - curtail_neg: Whether to curtail at negative prices
            - turbine_type: Wind turbine model (if tech='Wind')
        n_iterations: Number of Monte Carlo runs
        weather_years: List of years to sample for weather (default: 2005-2024)
        price_years: List of years to sample for prices (default: 2020-2026)
        df_market_hub: Pre-loaded market data (optional, for performance)
        progress_callback: Function to call with progress updates
        
    Returns:
        results_df: DataFrame with all iteration results
        stats: Dictionary with statistical summaries (P10, P50, P90, etc.)
    """
    
    # Default year ranges
    if weather_years is None:
        weather_years = list(range(2005, 2025))  # 2005-2024 (20 years)
    if price_years is None:
        price_years = list(range(2020, 2027))  # 2020-2026 (7 years)
    
    # Extract config
    tech = scenario_config.get('tech', 'Solar')
    capacity_mw = scenario_config.get('capacity_mw', 100.0)
    lat = scenario_config.get('lat', 32.0)
    lon = scenario_config.get('lon', -100.0)
    vppa_price = scenario_config.get('vppa_price', 50.0)
    revenue_share_pct = scenario_config.get('revenue_share', 100)
    curtail_neg = scenario_config.get('curtail_neg', False)
    turbine_type = scenario_config.get('turbine_type', 'GENERIC')
    
    revenue_share = revenue_share_pct / 100.0
    
    results = []
    errors = []
    
    for i in range(n_iterations):
        if progress_callback and i % 50 == 0:
            progress_callback(i, n_iterations)
        
        try:
            # 1. Random sample: weather year
            weather_year = random.choice(weather_years)
            
            # 2. Random sample: price year
            price_year = random.choice(price_years)
            
            # 3. Get generation profile for sampled weather year (from cache if available)
            if generation_profile_cache and weather_year in generation_profile_cache:
                generation_profile = generation_profile_cache[weather_year]
            else:
                # Fallback: fetch if not cached (slower)
                generation_profile = fetch_tmy.get_profile_for_year(
                    year=weather_year,
                    tech=tech,
                    lat=lat,
                    lon=lon,
                    capacity_mw=capacity_mw,
                    force_tmy=False,  # Use actual historical weather
                    turbine_type=turbine_type,
                    efficiency=0.86  # 14% losses
                )
            
            if generation_profile is None or generation_profile.empty:
                continue
            
            # Convert to Central Time
            gen_central = generation_profile.tz_convert('US/Central')
            
            # 4. Create generation DataFrame
            gen_df = pd.DataFrame({
                'Gen_MW': gen_central.values,
                'Time_Source': gen_central.index
            })
            
            # 5. Shift timestamps to price year for alignment
            def replace_year(ts):
                try:
                    return ts.replace(year=price_year)
                except ValueError:
                    # Handle Feb 29 in non-leap year
                    return ts + pd.DateOffset(days=1)
            
            gen_df['Time_Central'] = gen_df['Time_Source'].apply(replace_year)
            gen_df['Gen_Energy_MWh'] = gen_df['Gen_MW'] * 0.25  # 15-min intervals
            
            # 6. Load price data for sampled price year from cache
            if price_data_cache and price_year in price_data_cache:
                price_df = price_data_cache[price_year]
            else:
                print(f"Warning: No cached price data for year {price_year}")
                continue
            
            if price_df.empty:
                print(f"Warning: No price data found for year {price_year}")
                continue
            
            # 7. Merge generation with prices
            merged = pd.merge(
                price_df,
                gen_df[['Time_Central', 'Gen_Energy_MWh', 'Gen_MW']],
                on='Time_Central',
                how='inner'
            )
            
            if merged.empty:
                continue
            
            # 8. Apply curtailment if enabled
            mask_neg_price = merged['SPP'] < 0
            if curtail_neg:
                merged.loc[mask_neg_price, 'Gen_Energy_MWh'] = 0
            
            # 9. Calculate settlement using revenue share logic
            price_diff = merged['SPP'] - vppa_price
            
            if revenue_share < 1.0:
                # Asymmetric: Buyer gets revenue_share% of upside, 100% of downside
                settlement_per_mwh = (
                    np.maximum(price_diff, 0) * revenue_share +
                    np.minimum(price_diff, 0)
                )
            else:
                # Symmetric: Buyer gets 100% of both upside and downside
                settlement_per_mwh = price_diff
            
            merged['Settlement_$'] = merged['Gen_Energy_MWh'] * settlement_per_mwh
            merged['Market_Revenue_$'] = merged['Gen_Energy_MWh'] * merged['SPP']
            merged['VPPA_Payment_$'] = merged['Gen_Energy_MWh'] * vppa_price
            
            # 10. Calculate annual totals
            annual_settlement = merged['Settlement_$'].sum()
            annual_generation = merged['Gen_Energy_MWh'].sum()
            annual_market_revenue = merged['Market_Revenue_$'].sum()
            annual_vppa_payment = merged['VPPA_Payment_$'].sum()
            
            # 11. Store results
            results.append({
                'iteration': i,
                'weather_year': weather_year,
                'price_year': price_year,
                'annual_settlement_$': annual_settlement,
                'annual_generation_mwh': annual_generation,
                'annual_market_revenue_$': annual_market_revenue,
                'annual_vppa_payment_$': annual_vppa_payment,
                'avg_settlement_$/mwh': annual_settlement / annual_generation if annual_generation > 0 else 0
            })
            
        except Exception as e:
            # Track failed iterations for debugging
            error_msg = f"Iteration {i} (weather={weather_year}, price={price_year}): {str(e)}"
            print(error_msg)
            if i < 5:  # Only store first 5 errors to avoid memory issues
                errors.append(error_msg)
            continue
    
    if not results:
        print(f"ERROR: All {n_iterations} iterations failed!")
        print(f"Sample errors: {errors[:3]}")
        return pd.DataFrame(), {}
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistical summary
    settlement_values = results_df['annual_settlement_$']
    
    stats = {
        'P10': np.percentile(settlement_values, 10),   # Conservative (10th percentile)
        'P25': np.percentile(settlement_values, 25),
        'P50': np.percentile(settlement_values, 50),   # Median
        'P75': np.percentile(settlement_values, 75),
        'P90': np.percentile(settlement_values, 90),   # Optimistic (90th percentile)
        'Mean': np.mean(settlement_values),
        'StdDev': np.std(settlement_values),
        'Min': np.min(settlement_values),
        'Max': np.max(settlement_values),
        'Count': len(settlement_values)
    }
    
    return results_df, stats


def format_percentile_table(stats: Dict) -> pd.DataFrame:
    """
    Format statistical summary as a display-friendly DataFrame.
    """
    table_data = {
        'Percentile': ['P10 (Conservative)', 'P25', 'P50 (Median)', 'P75', 'P90 (Optimistic)'],
        'Annual Settlement ($)': [
            f"${stats['P10']:,.0f}",
            f"${stats['P25']:,.0f}",
            f"${stats['P50']:,.0f}",
            f"${stats['P75']:,.0f}",
            f"${stats['P90']:,.0f}"
        ]
    }
    
    return pd.DataFrame(table_data)


def compare_scenarios_monte_carlo(scenario_results: Dict[str, Tuple[pd.DataFrame, Dict]]) -> pd.DataFrame:
    """
    Compare Monte Carlo results across multiple scenarios.
    
    Args:
        scenario_results: Dict mapping scenario names to (results_df, stats) tuples
        
    Returns:
        Comparison DataFrame with P10/P50/P90 for each scenario
    """
    comparison_data = []
    
    for scenario_name, (results_df, stats) in scenario_results.items():
        # Skip scenarios with no results
        if not stats or len(stats) == 0:
            print(f"Warning: No statistics available for scenario '{scenario_name}'")
            continue
        
        # Safely get stats with defaults
        comparison_data.append({
            'Scenario': scenario_name,
            'P10 ($)': stats.get('P10', 0),
            'P50 ($)': stats.get('P50', 0),
            'P90 ($)': stats.get('P90', 0),
            'Mean ($)': stats.get('Mean', 0),
            'Std Dev ($)': stats.get('StdDev', 0),
            'P90-P10 Range ($)': stats.get('P90', 0) - stats.get('P10', 0)
        })
    
    return pd.DataFrame(comparison_data)
