import gridstatus
import pandas as pd

def check_capacity():
    iso = gridstatus.Ercot()
    target_date = "2024-11-01"
    print(f"Fetching SCED disclosure for {target_date}...")
    
    try:
        data = iso.get_60_day_sced_disclosure(date=target_date)
        if 'sced_gen_resource' in data:
            df = data['sced_gen_resource']
            print("\n--- Columns ---")
            print(df.columns.tolist())
            
            # Filter for VORTEX
            vortex = df[df['Resource Name'].str.contains("VORTEX_WIND", na=False)]
            
            if not vortex.empty:
                print("\n--- VORTEX_WIND Peak Capacity (HSL) ---")
                max_hsl = vortex.groupby('Resource Name')['HSL'].max()
                print(max_hsl)
                print(f"Total Peak HSL: {max_hsl.sum():.2f} MW")
            else:
                print("No VORTEX_WIND resources found.")

            # Also check AZURE_SOLAR
            azure = df[df['Resource Name'].str.contains("AZURE_SOLAR", na=False)]
            if not azure.empty:
                print("\n--- AZURE_SOLAR Peak Capacity (HSL) ---")
                max_hsl = azure.groupby('Resource Name')['HSL'].max()
                print(max_hsl)
                print(f"Total Peak HSL: {max_hsl.sum():.2f} MW")

        else:
            print("No 'sced_gen_resource' table in data.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_capacity()
