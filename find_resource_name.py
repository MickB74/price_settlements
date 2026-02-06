import gridstatus
import pandas as pd
from datetime import datetime, timedelta

def find_resource():
    iso = gridstatus.Ercot()
    # Pick a date definitely available (>60 days ago)
    target_date = "2024-11-01"
    print(f"Fetching SCED disclosure for {target_date}...")
    
    try:
        data = iso.get_60_day_sced_disclosure(date=target_date)
        if 'sced_gen_resource' in data:
            df = data['sced_gen_resource']
            print(f"Total Resources found: {len(df['Resource Name'].unique())}")
            
            # Search for patterns
            queries = ["LOS", "MIR", "HIDALGO", "RANCH", "MONTE"]
            
            print("\n--- Potential Matches ---")
            unique_names = sorted(df['Resource Name'].unique())
            
            found = False
            for name in unique_names:
                name_upper = str(name).upper()
                if any(q in name_upper for q in queries):
                    print(f"Found: {name}")
                    found = True
            
            if not found:
                print("No matches found for queries.")
                
        else:
            print("No 'sced_gen_resource' table in data.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_resource()
