import pandas as pd
import sys
import os
import datetime

# Add parent dir to path to import fetch_tmy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch_tmy

def main():
    # 1. Load Data
    input_file = 'ercot lat lon.xlsx'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file)
    
    # 2. Filter Solar
    # Check column names closely. Based on previous `head()` output: 'Technology'
    if 'Technology' not in df.columns:
        print("Error: 'Technology' column not found.")
        print(df.columns)
        return

    solar_df = df[df['Technology'].astype(str).str.contains('Solar', case=False, na=False)]
    
    results = []
    
    print(f"Found {len(solar_df)} solar projects.")
    
    # 3. Process Each Project
    for index, row in solar_df.iterrows():
        name = row.get('Project Name', f"Project {index}")
        # Assuming 'Project Latitude' and 'Project Longitude' are the column names from previous inspection
        lat = row.get('Project Latitude')
        lon = row.get('Project Longitude')
        cap = row.get('Nameplate Capacity(MWac)')
        
        # Skip invalid locations
        if pd.isna(lat) or pd.isna(lon):
            print(f"Skipping {name} due to missing coordinates.")
            continue
            
        # Default capacity if missing (though for total MWh output, capacity is critical)
        if pd.isna(cap) or cap == 0:
            print(f"Skipping {name} due to missing capacity.")
            continue

        print(f"Processing {name} ({cap} MW) at {lat}, {lon}...")
        
        try:
            # Fetch Profile (Use TMY for 2025 like the website)
            # The website uses fetch_tmy.get_profile_for_year
            # We used year=2025 in the plan.
            profile = fetch_tmy.get_profile_for_year(
                year=2025, 
                tech='Solar', 
                lat=lat, 
                lon=lon, 
                capacity_mw=cap, 
                force_tmy=True,
                efficiency=0.86  # 14% losses to match App default
            )
            
            if profile is not None:
                # profile is a pandas Series with DateTimeIndex (tz-aware usually UTC) and values in MW
                # Convert to dataframe for resampling
                pdf = pd.DataFrame({'MW': profile})
                
                # Check frequency to determine integration factor for Energy (MWh)
                # If hourly: sum. If 15-min: sum/4.
                # fetch_tmy normally aligns to hourly or 15-min based on source
                
                # Simple check:
                # If len is ~8760 -> hourly -> factor 1
                # If len is ~35040 -> 15min -> factor 0.25
                n_points = len(pdf)
                if n_points > 30000:
                    factor = 0.25
                else:
                    factor = 1.0
                
                # Resample to Month End ('ME') and sum
                monthly_mw = pdf.resample('ME')['MW'].sum() * factor
                
                # Build Row
                row_data = {
                    'Project Name': name,
                    'Project ID': row.get('Project ID'),
                    'Capacity (MW)': cap,
                    'Latitude': lat,
                    'Longitude': lon
                }
                
                # Add monthly data
                # We expect 12 months for 2025
                total_gen = 0
                for date, val in monthly_mw.items():
                    month_name = date.strftime('%B') # January, February...
                    row_data[month_name] = val
                    total_gen += val
                    
                row_data['Annual Total (MWh)'] = total_gen
                
                # Basic yield check
                if cap > 0:
                    row_data['Yield (MWh/MW)'] = total_gen / cap
                
                results.append(row_data)
            else:
                print(f"Failed to fetch profile for {name}")

        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

    # 4. Save
    if results:
        res_df = pd.DataFrame(results)
        output_file = 'solar_monthly_output_2025.csv'
        res_df.to_csv(output_file, index=False)
        print(f"Done. Processed {len(res_df)} projects. Saved to {output_file}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
