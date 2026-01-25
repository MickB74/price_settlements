import json
import os

def enrich_wind_registry():
    path = 'ercot_assets.json'
    if not os.path.exists(path):
        assets = {}
    else:
        with open(path, 'r') as f:
            assets = json.load(f)
            
    # New Wind Data (15+ Assets)
    new_data = {
        "Rio Bravo Wind": {
            "resource_name": "CABEZON_WIND1",
            "tech": "Wind",
            "capacity_mw": 237.6,
            "hub": "South",
            "lat": 26.353,
            "lon": -98.216,
            "county": "Starr",
            "hub_height_m": 105.0,
            "turbine_model": "Vestas V136-3.6"
        },
        "Route 66 Wind": {
            "resource_name": "ROUTE_66_WIND1",
            "tech": "Wind",
            "capacity_mw": 150.0,
            "hub": "Pan",
            "lat": 35.222,
            "lon": -101.831,
            "county": "Armstrong",
            "hub_height_m": 80.0,
            "turbine_model": "Vestas V110-2.0"
        },
        "Capricorn Ridge": {
            "resource_name": "CAPRIDG4_BB_PV", # SCED name for Capricorn Ph 4 usually
            "tech": "Wind",
            "capacity_mw": 663.0,
            "hub": "West",
            "lat": 31.85,
            "lon": -101.10,
            "county": "Sterling",
            "hub_height_m": 80.0,
            "turbine_model": "GE 1.5sle"
        },
        "Horse Hollow": {
            "resource_name": "HHOLLOW2_WIND1",
            "tech": "Wind",
            "capacity_mw": 735.5,
            "hub": "West",
            "lat": 32.32,
            "lon": -100.22,
            "county": "Taylor",
            "hub_height_m": 82.0,
            "turbine_model": "GE 1.5sle"
        },
        "South Plains": {
            "resource_name": "SPLAIN1_WIND1",
            "tech": "Wind",
            "capacity_mw": 500.0,
            "hub": "Pan",
            "lat": 34.18,
            "lon": -101.35,
            "county": "Floyd",
            "hub_height_m": 86.0,
            "turbine_model": "Vestas V117"
        },
        "Green Pastures": {
            "resource_name": "GPASTURE_WIND_I",
            "tech": "Wind",
            "capacity_mw": 300.0,
            "hub": "North",
            "lat": 33.65,
            "lon": -99.25,
            "county": "Baylor",
            "hub_height_m": 92.0,
            "turbine_model": "Acciona AW-116"
        },
        "Hereford Wind": {
            "resource_name": "HRFDWIND_WIND_G",
            "tech": "Wind",
            "capacity_mw": 200.0,
            "hub": "Pan",
            "lat": 34.82,
            "lon": -102.40,
            "county": "Deaf Smith",
            "hub_height_m": 88.0,
            "turbine_model": "Vestas V100"
        },
        "Cameron Wind": {
            "resource_name": "CAMWIND_UNIT1",
            "tech": "Wind",
            "capacity_mw": 165.0,
            "hub": "South",
            "lat": 26.15,
            "lon": -97.50,
            "county": "Cameron",
            "hub_height_m": 87.5,
            "turbine_model": "Nordex AW-125"
        },
        "San Roman Wind": {
            "resource_name": "SANROMAN_WIND_1",
            "tech": "Wind",
            "capacity_mw": 93.0,
            "hub": "South",
            "lat": 26.05,
            "lon": -97.40,
            "county": "Cameron",
            "hub_height_m": 87.5,
            "turbine_model": "Nordex AW125"
        },
        "Tyler Bluff": {
            "resource_name": "TYLRWIND_UNIT1",
            "tech": "Wind",
            "capacity_mw": 120.0,
            "hub": "North",
            "lat": 33.65,
            "lon": -97.20,
            "county": "Cooke",
            "hub_height_m": 80.0,
            "turbine_model": "Siemens SWT-2.3"
        },
        "Flat Top Wind": {
            "resource_name": "FTWIND_UNIT_1",
            "tech": "Wind",
            "capacity_mw": 200.0,
            "hub": "North",
            "lat": 31.75,
            "lon": -98.60,
            "county": "Mills",
            "hub_height_m": 80.0,
            "turbine_model": "GE 2.5-116"
        },
         "Bobcat Wind": {
            "resource_name": "BCATWIND_WIND_1",
            "tech": "Wind",
            "capacity_mw": 150.0,
            "hub": "West",
            "lat": 31.90,
            "lon": -101.40,
            "county": "Glasscock",
            "hub_height_m": 80.0,
            "turbine_model": "Mitsubishi MWT"
        },
        "Goat Mountain": {
            "resource_name": "GOAT_GOATWIND",
            "tech": "Wind",
            "capacity_mw": 150.0,
            "hub": "West",
            "lat": 32.00,
            "lon": -101.70,
            "county": "Sterling",
            "hub_height_m": 80.0,
            "turbine_model": "Mitsubishi 1.0"
        },
        "South Ranch Wind": {
            "resource_name": "SRWE1_SRWE2",
            "tech": "Wind",
            "capacity_mw": 100.0,
            "hub": "South",
            "lat": 26.50,
            "lon": -98.50,
            "county": "Hidalgo",
            "hub_height_m": 80.0,
            "turbine_model": "GE 2.x"
        },
        "Vera Wind": {
            "resource_name": "VERAWIND_UNIT1",
            "tech": "Wind",
            "capacity_mw": 240.0,
            "hub": "North",
            "lat": 33.60,
            "lon": -99.40,
            "county": "Knox",
            "hub_height_m": 80.0,
            "turbine_model": "GE 1.7"
        }
    }
    
    # Overwrite/Update
    for name, data in new_data.items():
        assets[name] = data
        
    with open(path, 'w') as f:
        json.dump(assets, f, indent=2)
        
    print(f"Successfully enriched registry with {len(new_data)} high-quality wind assets.")
    print(f"Total assets in registry: {len(assets)}")

if __name__ == "__main__":
    enrich_wind_registry()
