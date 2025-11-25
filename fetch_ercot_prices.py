import gridstatus
import pandas as pd
import patch_gridstatus # Apply monkey patch
import numpy as np
from datetime import datetime, timedelta

def fetch_ercot_north_prices(target_date=None, target_year=None, gen_type="solar", end_date=None):
    try:
        # Initialize ERCOT ISO
        iso = gridstatus.Ercot()

        # Determine mode
        if target_year:
            print(f"Running in Full Year Mode for {target_year}")
            year = int(target_year)
            target_date = None # Disable single day filtering
        else:
            # Determine date to fetch
            if target_date:
                if isinstance(target_date, str):
                    target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
            else:
                target_date = datetime.now().date()
                year = target_date.year

        # --- Day-Ahead Market (DAM) ---
        # Only fetch DAM if we are looking for a specific date (original functionality)
        if target_date and not target_year:
            print(f"Fetching ERCOT North Hub prices for {target_date}...")
            try:
                dam_spp = iso.get_spp(date=target_date, market="DAY_AHEAD_HOURLY")
                north_hub_dam = dam_spp[dam_spp["Location"] == "HB_NORTH"]
                
                if not north_hub_dam.empty:
                    print("ERCOT North Hub (HB_NORTH) Day-Ahead Prices (First 5 rows):")
                    print(north_hub_dam[["Time", "Location", "Market", "SPP"]].head().to_string(index=False))
                    
                    filename = "ercot_north_dam_prices.csv"
                    north_hub_dam.to_csv(filename, index=False)
                    print(f"Data saved to {filename}")
                else:
                    print("No data found for HB_NORTH on this date.")
            except Exception as e:
                print(f"Error fetching DAM prices: {e}")

        # --- Real-Time Market (RTM) ---
        print("\n--- Real-Time Market (RTM) ---")
        
        # Fetch RTM data for the year (contains all hubs)
        # We fetch once and then filter
        try:
            if target_year:
                print(f"Fetching RTM data for year {year} (this may take a moment)...")
                rtm_spp = iso.get_rtm_spp(year=year)
            else:
                print(f"Fetching RTM data for year {year} (this may take a moment)...")
                rtm_spp = iso.get_rtm_spp(year=year)
                
            print(f"Fetched {len(rtm_spp)} rows for {year}.")
            
            # Define Hubs
            hubs = ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON"]
            
            for hub in hubs:
                print(f"\nProcessing {hub}...")
                
                # Filter for Hub
                hub_rtm = rtm_spp[rtm_spp["Location"] == hub].copy()
                
                # Ensure 'Time' column is datetime
                if not pd.api.types.is_datetime64_any_dtype(hub_rtm['Time']):
                     hub_rtm['Time'] = pd.to_datetime(hub_rtm['Time'], utc=True)
                
                # Create Central Time column for filtering
                hub_rtm['Time_Central'] = hub_rtm['Time'].dt.tz_convert('US/Central')

                if target_date:
                    # Filter for specific date
                    hub_rtm = hub_rtm[hub_rtm['Time_Central'].dt.date == target_date]
                
                if end_date:
                    # Filter by end date (inclusive)
                    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
                    hub_rtm = hub_rtm[hub_rtm['Time_Central'].dt.date <= end_date_obj]
                    print(f"Filtered data through {end_date_obj}.")
                
                if not hub_rtm.empty:

                    # --- VPPA Settlement Calculation ---
                    strike_price = 30.0
                    peak_mw = 80.0
                    interval_hours = 0.25 # 15 minutes
                    
                    # Calculate Settlement Price ($/MWh)
                    hub_rtm['Strike_Price'] = strike_price
                    hub_rtm['Settlement_Price'] = hub_rtm['SPP'] - strike_price
                    
                    # Calculate Energy Quantity (MWh)
                    
                    if gen_type == 'solar':
                        import solar_profile
                        
                        def calculate_solar_mw_point(dt, peak):
                            day_of_year = dt.timetuple().tm_yday
                            day_length = 12 + 2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                            solar_noon = 13.5
                            sunrise = solar_noon - (day_length / 2)
                            sunset = solar_noon + (day_length / 2)
                            t = dt.hour + dt.minute / 60.0
                            
                            if t < sunrise or t > sunset:
                                return 0.0
                            else:
                                return peak * np.sin(np.pi * (t - sunrise) / day_length)

                        # Apply the calculation
                        hub_rtm['Potential_Gen_MW'] = hub_rtm['Time_Central'].apply(lambda x: calculate_solar_mw_point(x, peak_mw))
                        gen_col_name = "Solar_MW"
                        
                    elif gen_type == 'wind':
                        import wind_profile
                        # For wind, the profile generator returns a full series. 
                        # But we are processing row by row or using apply.
                        # The wind profile generator uses random noise seeded by date.
                        # We need to ensure consistency.
                        # Let's use a similar point-based calculation or generate the full profile and map it.
                        # Since wind profile has noise, point-based is tricky if we want continuity.
                        # BUT, for this synthetic exercise, calculating point-based with a deterministic seed (based on hour/date) is fine.
                        
                        def calculate_wind_mw_point(dt, peak):
                            # Re-implement logic from wind_profile.py for single point
                            t = dt.hour + dt.minute / 60.0
                            
                            # Base signal
                            base_signal = np.sin((t + 4) * 2 * np.pi / 24)
                            normalized_base = (base_signal + 1) / 2
                            scaled_base = 0.2 + 0.6 * normalized_base
                            
                            # Noise: Deterministic based on time to ensure consistency across calls for same row
                            # Using a simple sine-based noise for deterministic behavior in 'apply'
                            noise = 0.15 * np.sin(t * 10) 
                            
                            profile = scaled_base + noise
                            profile = np.clip(profile, 0, 1)
                            return profile * peak

                        hub_rtm['Potential_Gen_MW'] = hub_rtm['Time_Central'].apply(lambda x: calculate_wind_mw_point(x, peak_mw))
                        gen_col_name = "Wind_MW"

                    
                    # Economic Curtailment: If SPP < 0, Actual Gen MW = 0
                    # User correction: "i only curtail when market ibelow 0"
                    hub_rtm['Actual_Gen_MW'] = np.where(hub_rtm['SPP'] < 0, 0.0, hub_rtm['Potential_Gen_MW'])
                    
                    # Calculate Energy
                    hub_rtm['Gen_Energy_MWh'] = hub_rtm['Actual_Gen_MW'] * interval_hours
                    hub_rtm['Curtailed_MWh'] = (hub_rtm['Potential_Gen_MW'] - hub_rtm['Actual_Gen_MW']) * interval_hours

                    # Calculate Settlement Amount ($)
                    hub_rtm['Settlement_Amount'] = hub_rtm['Settlement_Price'] * hub_rtm['Gen_Energy_MWh']
                    
                    # Calculate Cumulative Settlement
                    hub_rtm['Cumulative_Settlement'] = hub_rtm['Settlement_Amount'].cumsum()
                    
                    # Calculate Net Settlement
                    net_settlement = hub_rtm['Settlement_Amount'].sum()
                    total_energy = hub_rtm['Gen_Energy_MWh'].sum()
                    total_curtailed = hub_rtm['Curtailed_MWh'].sum()
                    
                    print(f"VPPA Settlement for {hub} ({gen_type.capitalize()}, Strike: ${strike_price}/MWh, Peak: {peak_mw} MW):")
                    if target_date:
                        print(f"Net Settlement for {target_date}: ${net_settlement:,.2f}")
                        print(f"Total Energy Generated: {total_energy:,.2f} MWh")
                        print(f"Total Energy Curtailed: {total_curtailed:,.2f} MWh")
                        filename_vppa = f"vppa_settlement_{hub}_{gen_type}.csv"
                    else:
                        print(f"Net Settlement for Year {year}: ${net_settlement:,.2f}")
                        print(f"Total Energy Generated: {total_energy:,.2f} MWh")
                        print(f"Total Energy Curtailed: {total_curtailed:,.2f} MWh")
                        filename_vppa = f"vppa_settlement_{year}_{hub}_{gen_type}.csv"
                    
                    # Save Settlement Data
                    columns_vppa = [
                        "Time", "Location", "SPP", "Strike_Price", "Settlement_Price", 
                        "Potential_Gen_MW", "Actual_Gen_MW", "Curtailed_MWh", 
                        "Gen_Energy_MWh", "Settlement_Amount", "Cumulative_Settlement"
                    ]
                    hub_rtm[columns_vppa].to_csv(filename_vppa, index=False)
                    print(f"VPPA Settlement data saved to {filename_vppa}")
                    
                else:
                    print(f"No RTM data found for {hub}.")

        except Exception as e:
            print(f"Error fetching RTM prices: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch ERCOT prices and calculate VPPA settlement.")
    parser.add_argument("--year", type=int, default=2024, help="Year to analyze (default: 2024)")
    parser.add_argument("--type", type=str, default="solar", choices=["solar", "wind"], help="Generation type (solar or wind)")
    parser.add_argument("--end-date", type=str, help="End date for filtering (YYYY-MM-DD)")
    args = parser.parse_args()
    
    fetch_ercot_north_prices(target_year=args.year, gen_type=args.type, end_date=args.end_date)
