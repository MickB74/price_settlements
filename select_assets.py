import gridstatus
import pandas as pd
from datetime import datetime, timedelta

def select_assets():
    iso = gridstatus.Ercot()
    target_date = datetime.now() - timedelta(days=65)
    print(f"Fetching data for {target_date.date()}...")
    
    try:
        data = iso.get_60_day_sced_disclosure(date=target_date)
        df_gen = data['sced_gen_resource']
        
        # Strip spaces from columns
        df_gen.columns = [c.strip() for c in df_gen.columns]
        
        # Filter for Solar and Wind
        solar = df_gen[df_gen['Resource Type'] == 'PV'] # Usually PV for solar
        wind = df_gen[df_gen['Resource Type'] == 'WTR'] # Usually WTR for wind? Wait, let's check types.
        
        print("\nUnique Resource Types:")
        print(df_gen['Resource Type'].unique())
        
        # Actually let's look for common ones
        # WIND = 'WSR' (Wind-powered Generation Resource) or 'WTR'?
        # SOLAR = 'PV' or 'GR'?
        
        # Let's just find the ones with most generation
        top_gen = df_gen.groupby(['Resource Name', 'Resource Type'])['Telemetered Net Output'].max().sort_values(ascending=False).head(50)
        print("\nTop Generation by Resource:")
        print(top_gen)
        
        # Filter again with better type knowledge
        # Actually it seems Resource Type might be descriptive.
        
        # Let's save a broad sample for the user to see
        sample_assets = []
        for rtype in ['PV', 'WSR', 'GR']: # Guessing
            subset = df_gen[df_gen['Resource Type'] == rtype]
            if not subset.empty:
                names = subset['Resource Name'].unique()[:10]
                for name in names:
                    sample_assets.append({"name": name, "type": rtype, "max_mw": subset[subset['Resource Name'] == name]['Telemetered Net Output'].max()})
        
        print("\nSample Assets Found:")
        for a in sample_assets:
            print(f"{a['name']} ({a['type']}): {a['max_mw']} MW")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    select_assets()
