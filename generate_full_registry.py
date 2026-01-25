import gridstatus
import pandas as pd
import json
import re
from geopy.geocoders import Nominatim
import time

# Dictionary of Known Mappings (Manual Overrides for tricky ones)
KNOWN_MAPPINGS = {
    "ADL_SOLAR1": "Adelanto Solar",
    "ANCHOR_WIND": "Anchor Wind",
    "AZSP_SLR": "Azalea Springs",
    "BAKERSFIELD": "Bakersfield Solar",
    "BART_SLR": "Bart Solar",
    "BCATWIND": "Bobcat Wind",
    "BLVN_SLR": "Bluebonnet Solar",
    "BRISCOE": "Briscoe Wind",
    "BYNM_SLR": "Bynum Solar",
    "CABEZON": "Cabezon Wind",
    "CAMWIND": "Cameron Wind",
    "CAPRIDG": "Capricorn Ridge",
    "CHAL_SLR": "Chariot Solar",
    "CHIL_SLR": "Childress Solar",
    "CMPD_SLR": "Camp Solar",
    "CORALSLR": "Coral Solar",
    "CTW_SOLAR": "Cottonwood Solar",
    "DIVR_SLR": "Diver Solar",
    "DMA": "Dumas Solar",
    "DORA_SLR": "Dora Solar",
    "DRCK_SLR": "Dry Creek Solar",
    "ELZA_SLR": "Eliza Solar",
    "ESTONIAN": "Estonian Solar",
    "EUNICE": "Eunice Wind",
    "EXGNSND": "Exgen Sound",
    "EXGNWTL": "Exgen Whitetail",
    "FENCESLR": "Fence Post Solar",
    "FERMI": "Fermi Wind",
    "FILESSLR": "Files Solar",
    "FRYE_SLR": "Frye Solar",
    "FTWIND": "Flat Top Wind",
    "GAIA_SLR": "Gaia Solar",
    "GALLOWAY": "Galloway Solar",
    "GOAT": "Goat Wind",
    "GPASTURE": "Green Pasture",
    "GRIZZLY": "Grizzly Solar",
    "GRYH_SLR": "Greyhound Solar",
    "HHOLLOW": "Horse Hollow",
    "HOLSTEIN": "Holstein Solar",
    "HRFDWIND": "Hereford Wind",
    "JKLP_SLR": "Blue Jay Solar",
    "LILY": "Lily Solar",
    "LMWD_SLR": "Lakewood Solar",
    "LNP": "Long Point",
    "LON": "Lion Solar",
    "MERCURY": "Mercury Solar",
    "MIDP_SLR": "Midpoint Solar",
    "MIDWIND": "Midway Wind",
    "MLB_SLR": "Mockingbird Solar",
    "MONTECR": "Monte Cristo",
    "MOZART": "Mozart Wind",
    "MRKM_SLR": "Markham Solar",
    "MROW_SLR": "Maryneal",
    "MUSTNGCK": "Mustang Creek",
    "NOBLESLR": "Noble Solar",
    "NRTN_SLR": "Norton Solar",
    "PALMWIND": "Palmas Wind",
    "PHO": "Pho Solar",
    "PISGAH": "Pisgah Solar",
    "QUEEN_SL": "Queen Solar",
    "RATLIFF": "Ratliff Solar",
    "ROSELAND": "Roseland Solar",
    "ROUTE_66": "Route 66 Wind",
    "RRC_WIND": "Roadrunner",
    "SANROMAN": "San Roman",
    "SOLARA": "Solara",
    "SPLAIN": "South Plains",
    "SRWE": "South Ranch",
    "SSPUR": "Silver Spur",
    "STAM_SLR": "Stamford Solar",
    "STLHS_SL": "Stillhouse Solar",
    "TI_SOLAR": "Taygete",
    "TNG_SOLAR": "Tango Solar",
    "TRBT_SLR": "Tributary Solar",
    "TREB_SLR": "Trebut Solar",
    "TROJ_SLR": "Trojan Solar",
    "TYLRWIND": "Tyler Bluff",
    "VERAWIND": "Vera Wind",
    "VERTIGO": "Vertigo Wind",
    "VORTEX": "Vortex Wind",
    "WHTTAIL": "Whitetail",
    "WH_WIND": "Whitehorse",
    "WILDWIND": "Wildwind",
    "ZIER_SLR": "Zier Solar",
    "BCATWIND": "Bobcat Wind",
    "BRISCOE": "Briscoe Wind",
    "CABEZON": "Rio Bravo Wind",
    "CAMWIND": "Cameron Wind",
    "CAPRIDG": "Capricorn Ridge",
    "FERMI": "Fermi Wind",
    "FTWIND": "Flat Top Wind",
    "GOAT": "Goat Mountain",
    "GPASTURE": "Green Pastures",
    "HHOLLOW": "Horse Hollow",
    "HRFDWIND": "Hereford Wind",
    "MOZART": "Mozart Wind",
    "PALMWIND": "Palmas Altas",
    "SANROMAN": "San Roman",
    "SPLAIN": "South Plains",
    "SRWE": "South Ranch",
    "SSPUR": "Silver Spur",
    "TYLRWIND": "Tyler Bluff",
    "VERAWIND": "Vera Wind",
    "VERTIGO": "Vertigo Wind",
    "VORTEX": "Vortex Wind",
    "SSPUR": "Spinning Spur",
    "ADL_SOLAR": "Adlong Solar",
    "ASCK_SLR": "Azalea Springs",
    "AZSP_SLR": "Azalea Springs",
    "AZURE_SOLAR": "Azure Sky",
    "BLVN_SLR": "Blevins Solar",
    "CHAL_SLR": "Chaluane Solar",
    "CHIL_SLR": "Chillicothe Solar",
    "CMPD_SLR": "Compound Solar",
    "CORALSLR": "Coral Solar",
    "CTW_SOLAR": "CTW Solar",
    "DMA": "DMA Solar",
    "ESTONIAN_SOLAR": "Estonian Solar",
    "EUNICE_PV": "2W Permian",
    "FENCESLR": "Fence Post",
    "FILESSLR": "Files Solar",
    "GALLOWAY": "Galloway Solar",
    "GRIZZLY": "Grizzly Solar",
    "MERCURY_PV": "Mercury Solar",
    "MRKM_SLR": "Markham Solar",
    "MUSTNGCK": "Mustang Creek",
    "NOBLESLR": "BT Noble",
    "PHO": "Phoenix Solar",
    "STLHS_SL": "Stellhaus Solar",
    "TI_SOLAR": "TI Solar",
    "TNG_SOLAR": "TNG Solar",
    "TRBT_SLR": "Trumbull Solar",
    "TREB_SLR": "Treeline Solar",
    "TROJ_SLR": "Trojan Solar"
}

# Technical Specs for Advanced Modeling (Hub Height, Turbine Type)
ADVANCED_SPECS = {
    "Rio Bravo Wind": {"hub_height_m": 105, "turbine_model": "Vestas V136-3.6"},
    "Route 66 Wind": {"hub_height_m": 80, "turbine_model": "Vestas V110-2.0"},
    "Capricorn Ridge": {"hub_height_m": 80, "turbine_model": "GE 1.5sle / Siemens SWT"},
    "Horse Hollow": {"hub_height_m": 82, "turbine_model": "GE 1.5sle / Siemens SWT"},
    "South Plains": {"hub_height_m": 86, "turbine_model": "Vestas V100/V117"},
    "Tyler Bluff": {"hub_height_m": 80, "turbine_model": "Siemens SWT-2.3-108"},
    "Green Pastures": {"hub_height_m": 92, "turbine_model": "Nordex AW-116/3000"},
    "Hereford Wind": {"hub_height_m": 88, "turbine_model": "GE 1.85 / Vestas V100"},
    "Briscoe Wind": {"hub_height_m": 80, "turbine_model": "GE 1.85-87"},
    "Cameron Wind": {"hub_height_m": 87.5, "turbine_model": "Nordex AW-125/3000"},
    "Palmas Altas": {"hub_height_m": 87.5, "turbine_model": "Nordex AW125/3150"},
    "San Roman": {"hub_height_m": 87.5, "turbine_model": "Nordex AW125/3000"},
    "Vortex Wind": {"hub_height_m": 80, "turbine_model": "GENERIC"}
}

def clean_name(name):
    """Simplifies names for better matching."""
    if not name: return ""
    name_upper = str(name).upper()
    
    # Check Manual Overrides first
    for k, v in KNOWN_MAPPINGS.items():
        if k in name_upper:
            return v.upper()

    # Fallback to heuristics
    name_clean = re.sub(r'(_UNIT\d+|_SOLAR\d+|_WIND\d+|_PV\d+|_SR\d+|_WR\d+)', '', name_upper)
    name_clean = re.sub(r'( UNIT \d+| SOLAR| WIND| PV| PROJECT)', '', name_clean)
    name_clean = name_clean.replace("_", " ") # Replace underscores with spaces
    return name_clean.strip()

def generate():
    iso = gridstatus.Ercot()
    
    # 1. Load SCED resource names
    with open('ercot_renewable_assets.txt', 'r') as f:
        sced_names = [line.strip() for line in f if line.strip()]
        
    # 2. Fetch Interconnection Queue
    print("Fetching Interconnection Queue...")
    queue = iso.get_interconnection_queue()
    # Filter for Completed/In Service
    queue = queue[queue['Status'].isin(['Completed', 'In Service', 'Synchronized'])]
    queue = queue[queue['Generation Type'].str.contains('Solar|Wind', case=False, na=False)]
    
    print(f"Queue size loaded: {len(queue)} projects")
    
    # 3. Try to match
    registry = {}
    geocoder = Nominatim(user_agent="ercot_mapper_v2")
    county_cache = {}

    print(f"Processing {len(sced_names)} SCED resources...")
    
    matches_found = 0
    
    for r_name in sced_names:
        search_term = clean_name(r_name)
        
        # Look for a match in queue
        if len(search_term) < 4: continue # Skip too short
        
        # Try finding the search term inside Project Name
        match = queue[queue['Project Name'].str.upper().str.contains(search_term, na=False, regex=False)].sort_values('Capacity (MW)', ascending=False)
        
        if not match.empty:
            matches_found += 1
            best = match.iloc[0]
            county = best['County']
            tech = "Solar" if "Solar" in str(best['Generation Type']) else "Wind"
            
            # Geocode County if not in cache (and valid county)
            if county and isinstance(county, str) and county not in county_cache:
                print(f"Geocoding {county}, TX...")
                try:
                    loc = geocoder.geocode(f"{county} County, TX")
                    if loc:
                        county_cache[county] = (loc.latitude, loc.longitude)
                    else:
                        county_cache[county] = (None, None)
                    time.sleep(1) 
                except:
                    county_cache[county] = (None, None)
            
            lat, lon = county_cache.get(county, (None, None))
            
            if lat:
                # Approximate Hub mapping
                hub = "North"
                if lon < -101: hub = "West"
                elif lat < 30: hub = "South"
                elif lon > -96: hub = "Houston"
                
                # Use Friendly Name
                friendly = best['Project Name'].title()
                
                registry[friendly] = {
                    "resource_name": r_name,
                    "project_name": friendly,
                    "tech": tech,
                    "capacity_mw": float(best['Capacity (MW)']),
                    "hub": hub,
                    "lat": lat,
                    "lon": lon,
                    "county": county
                }
                
                # Apply Advanced Specs if available
                if friendly in ADVANCED_SPECS:
                    registry[friendly].update(ADVANCED_SPECS[friendly])
                elif tech == "Wind":
                    # Add default wind specs if missing
                    registry[friendly].update({"hub_height_m": 80, "turbine_model": "GENERIC"})
        else:
            # print(f"No match for {r_name} (Search: {search_term})")
            pass

    # 4. Save
    print(f"Total Matches Found: {matches_found}")
    
    # Merge with existing to keep any manual edits
    try:
        with open('ercot_assets.json', 'r') as f:
            existing = json.load(f)
            registry.update(existing) # Keep existing ones, overwrite if collision
    except:
        pass

    with open('ercot_assets.json', 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"Updated registry now has {len(registry)} assets.")

if __name__ == "__main__":
    generate()
