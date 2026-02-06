import pandas as pd
import json
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r

def enrich_solar():
    print("Loading ercot_assets.json...")
    with open('ercot_assets.json', 'r') as f:
        assets = json.load(f)
        
    print("Loading USPVDB JSON...")
    try:
        with open('uspvdb_tx.json', 'r') as f:
            data = json.load(f)
            df_solar = pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    print(f"Loaded {len(df_solar)} solar projects.")
    
    # Filter not needed if API handled it, but safety check
    # keys: p_name, p_state, xlong, ylat, p_axis_no, p_cap_ac, p_cap_dc
    
    updated_count = 0
    
    for name, meta in assets.items():
        if meta.get('tech') != 'Solar':
            continue
            
        lat = meta.get('lat')
        lon = meta.get('lon')
        
        if not lat or not lon:
            continue
            
        # Spatial Match (5km radius approx 0.05 deg, widened to 0.1)
        lat_window = 0.1
        lon_window = 0.1
        
        candidates = df_solar[
            (df_solar['ylat'] > lat - lat_window) & 
            (df_solar['ylat'] < lat + lat_window) &
            (df_solar['xlong'] > lon - lon_window) &
            (df_solar['xlong'] < lon + lon_window)
        ].copy()
        
        if candidates.empty:
            # print(f"  No match for {name}")
            continue
            
        candidates['dist_km'] = candidates.apply(
            lambda row: haversine(lon, lat, row['xlong'], row['ylat']), axis=1
        )
        
        # Closest match within 5km (Solar farms are big)
        matches = candidates[candidates['dist_km'] < 5.0].sort_values('dist_km')
        
        if matches.empty:
             print(f"  No match <5km for {name}")
             continue
             
        best = matches.iloc[0]
        
        # Extract attributes from JSON keys
        # p_axis: "single-axis", "dual-axis", "fixed-tilt"
        axis_str = str(best.get('p_axis', '')).lower()
        
        tracking = False
        track_str = "Fixed Tilt"
        
        if "single" in axis_str or "dual" in axis_str: # Dual is tracking too
            tracking = True
            track_str = "Single-Axis"
        elif "fixed" in axis_str:
            track_str = "Fixed Tilt"
        else:
            track_str = f"Unknown({axis_str})"
            
        # Capacities
        cap_ac = float(best.get('p_cap_ac') or 0)
        cap_dc = float(best.get('p_cap_dc') or 0)
        
        dc_ac_ratio = 1.3 # Default
        if cap_ac > 0 and cap_dc > 0:
            dc_ac_ratio = cap_dc / cap_ac
            
        print(f"  Matched {name}: {track_str}, ILR={dc_ac_ratio:.2f} (USPVDB: {best.get('p_name')})")
        
        # Update Asset
        assets[name]['tracking_type'] = "single_axis" if tracking else "fixed"
        assets[name]['dc_ac_ratio'] = dc_ac_ratio
        # Store raw flags too if needed
        assets[name]['uspvdb_id'] = str(best.get('case_id'))
        
        updated_count += 1

    print(f"\nUpdated {updated_count} solar projects.")
    with open('ercot_assets.json', 'w') as f:
        json.dump(assets, f, indent=4)
    print("Saved to ercot_assets.json")

if __name__ == "__main__":
    enrich_solar()
