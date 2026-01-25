import pandas as pd
import json
import numpy as np
import os
from geopy.distance import geodesic

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_FILE = os.path.join(BASE_DIR, "ercot_assets.json")
USWTDB_FILE = os.path.join(BASE_DIR, "data_static", "uswtdb.csv")

def enrich_registry():
    print("--- enriching registry with USWTDB data ---")
    
    # Load Assets
    with open(ASSETS_FILE, 'r') as f:
        assets = json.load(f)
        
    # Load USWTDB
    if not os.path.exists(USWTDB_FILE):
        print("USWTDB file not found. Run fetch_metadata.py first.")
        return

    print("Loading US Wind Turbine Database...")
    df_turbines = pd.read_csv(USWTDB_FILE, low_memory=False)
    # Filter for Texas
    df_tx = df_turbines[df_turbines['t_state'] == 'TX'].copy()
    print(f"Loaded {len(df_tx)} turbines in Texas.")

    updated_count = 0
    
    for name, meta in assets.items():
        if meta.get('tech') != 'Wind':
            continue
            
        lat = meta.get('lat')
        lon = meta.get('lon')
        
        if not lat or not lon:
            continue
            
        # Spatial Match: Find turbines within 10km
        # Simple bounding box filter first for speed
        box_deg = 0.15 # approx 15km
        nearby = df_tx[
            (df_tx['ylat'] > lat - box_deg) & (df_tx['ylat'] < lat + box_deg) &
            (df_tx['xlong'] > lon - box_deg) & (df_tx['xlong'] < lon + box_deg)
        ].copy()
        
        if nearby.empty:
            print(f"Warning: No turbines found near {name} ({meta['resource_name']})")
            continue
            
        # Calculate precise distances
        origin = (lat, lon)
        nearby['dist_km'] = nearby.apply(lambda row: geodesic(origin, (row['ylat'], row['xlong'])).km, axis=1)
        
        # Filter to 10km radius
        cluster = nearby[nearby['dist_km'] <= 10.0]
        
        if cluster.empty:
             print(f"Warning: No turbines within 10km of {name}")
             continue
             
        # Find Mode (Most Common) Turbine Model
        # t_manu = Manufacturer, t_model = Model
        top_model = cluster['t_model'].mode()
        if top_model.empty:
             continue
        model_str = top_model[0]
        
        top_manu = cluster['t_manu'].mode()[0] if not cluster['t_manu'].mode().empty else "Unknown"
        
        # specs
        avg_height = cluster['t_hh'].mean()
        avg_rotor = cluster['t_rd'].mean()
        
        print(f"Match found for {name}: {top_manu} {model_str} (H={avg_height:.1f}m, D={avg_rotor:.1f}m)")
        
        # Update Metadata
        meta['turbine_manuf'] = top_manu
        meta['turbine_model'] = str(model_str)
        meta['hub_height_m'] = float(avg_height) if not pd.isna(avg_height) else None
        meta['rotor_diameter_m'] = float(avg_rotor) if not pd.isna(avg_rotor) else None
        
        updated_count += 1
        
    # Save updated registry
    if updated_count > 0:
        with open(ASSETS_FILE, 'w') as f:
            json.dump(assets, f, indent=2)
        print(f"\n✅ Updated {updated_count} wind assets in {ASSETS_FILE}")
        
if __name__ == "__main__":
    enrich_registry()
