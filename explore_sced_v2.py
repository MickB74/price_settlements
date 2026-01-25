import gridstatus
import pandas as pd
from datetime import datetime, timedelta

def explore_sced_details():
    iso = gridstatus.Ercot()
    target_date = datetime.now() - timedelta(days=65)
    print(f"Fetching 60-day SCED disclosure for {target_date.date()}...")
    
    try:
        data = iso.get_60_day_sced_disclosure(date=target_date)
        df_gen = data['sced_gen_resource']
        
        # Look for generation columns
        gen_cols = [c for c in df_gen.columns if any(x in c.lower() for x in ['mw', 'power', 'gen', 'base'])]
        print(f"\nPotential Generation Columns: {gen_cols}")
        
        # Look for type columns
        type_cols = [c for c in df_gen.columns if any(x in c.lower() for x in ['type', 'fuel', 'tech'])]
        print(f"Potential Type Columns: {type_cols}")
        
        # Print sample resources and their values
        # Common ERCOT names: _SOLAR, _WIND, _PV, _SR, _WR
        sample_resources = df_gen[df_gen['Resource Name'].str.contains('SOLAR|WIND|PV|_SR|_WR', case=False, na=False)]
        
        if not sample_resources.empty:
            print(f"\nFound {len(sample_resources['Resource Name'].unique())} likely renewable units.")
            print("\nSample Units with Generation Data:")
            display_cols = ['Interval Start', 'Resource Name'] + gen_cols[:3]
            print(sample_resources[display_cols].head(10))
            
            # Save a list of unique names to a file for later use
            unique_assets = sample_resources['Resource Name'].unique()
            with open('ercot_renewable_assets.txt', 'w') as f:
                for asset in sorted(unique_assets):
                    f.write(f"{asset}\n")
            print(f"\nSaved {len(unique_assets)} asset names to ercot_renewable_assets.txt")
        else:
            print("\nNo obvious renewable units found by name filter.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explore_sced_details()
