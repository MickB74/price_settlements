import gridstatus
import pandas as pd

def check_cols():
    iso = gridstatus.Ercot()
    target_date = "2024-11-01"
    print(f"Fetching SCED disclosure for {target_date}...")
    
    try:
        data = iso.get_60_day_sced_disclosure(date=target_date)
        if 'sced_gen_resource' in data:
            df = data['sced_gen_resource']
            print("--- Columns ---")
            print(df.columns.tolist())
            
            # Print a sample row
            print("--- Sample Row ---")
            print(df.head(1).T)
        else:
            print("No 'sced_gen_resource' table in data.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_cols()
