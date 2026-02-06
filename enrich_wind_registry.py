import pandas as pd
import json
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Haversine formula for distance
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def enrich_assets():
    # 1. Load Assets
    print("Loading ercot_assets.json...")
    with open('ercot_assets.json', 'r') as f:
        assets = json.load(f)
        
    # 2. Load USWTDB (JSON)
    print("Loading USWTDB JSON...")
    try:
        with open('uswtdb_tx.json', 'r') as f:
            data = json.load(f)
            df_turbines = pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    print(f"Loaded {len(df_turbines)} turbines.")
    
    # Filter for Texas (API already did this, but safe to keep)
    # API keys might differ slightly from CSV, typically they match
    # keys: t_state, xlong, ylat, t_hh, t_rd, t_manu, t_model
    if 't_state' in df_turbines.columns:
        df_tx = df_turbines[df_turbines['t_state'] == 'TX'].copy()
    else:
        df_tx = df_turbines.copy() # Assume it's already filtered
        
    print(f"Working with {len(df_tx)} turbines.")

    updated_count = 0
    
    # 3. Iterate Wind Projects
    for name, meta in assets.items():
        if meta.get('tech') != 'Wind':
            continue
            
        lat = meta.get('lat')
        lon = meta.get('lon')
        
        if not lat or not lon:
            print(f"Skipping {name} (No coords)")
            continue
            
        # 4. Find matches within 5km radius (projects are large)
        # Vectorized distance calc is faster but loop is fine for <50 projects
        # Let's do a quick box filter first
        lat_window = 0.1 # approx 10km
        lon_window = 0.1
        
        candidates = df_tx[
            (df_tx['ylat'] > lat - lat_window) & 
            (df_tx['ylat'] < lat + lat_window) &
            (df_tx['xlong'] > lon - lon_window) &
            (df_tx['xlong'] < lon + lon_window)
        ].copy()
        
        if candidates.empty:
            print(f"  No USWTDB match for {name}")
            continue
            
        # Precise distance
        candidates['dist_km'] = candidates.apply(
            lambda row: haversine(lon, lat, row['xlong'], row['ylat']), axis=1
        )
        
        # Keep closest cluster (e.g. within 3km)
        matches = candidates[candidates['dist_km'] < 3.0]
        
        if matches.empty:
            # Fallback: maybe the asset lat/lon is off-center. Try 10km.
            matches = candidates[candidates['dist_km'] < 8.0]
            
        if matches.empty:
             print(f"  No nearby turbines for {name}")
             continue
             
        # 5. Aggregate Specs
        # Hub height
        avg_hub = matches['t_hh'].mean()
        # Rotor Diameter
        avg_rotor = matches['t_rd'].mean()
        # Manufacturer (Mode)
        try:
            mode_manuf = matches['t_manu'].mode().iloc[0]
        except:
            mode_manuf = "GENERIC"
            
        # Model (Mode)
        try:
            mode_model = matches['t_model'].mode().iloc[0]
        except:
            mode_model = "GENERIC"
            
        print(f"  Matched {name}: {len(matches)} turbines. "
              f"Hub: {avg_hub:.1f}m, Rotor: {avg_rotor:.1f}m, Type: {mode_manuf} {mode_model}")
              
        # 6. Update Asset
        assets[name]['hub_height_m'] = float(avg_hub)
        assets[name]['rotor_diameter_m'] = float(avg_rotor)
        assets[name]['turbine_manuf'] = str(mode_manuf)
        assets[name]['turbine_model'] = str(mode_model)
        updated_count += 1

    # 7. Save
    print(f"\nUpdated {updated_count} wind projects.")
    with open('ercot_assets.json', 'w') as f:
        json.dump(assets, f, indent=4)
    print("Saved to ercot_assets.json")

if __name__ == "__main__":
    enrich_assets()
