import pandas as pd
import glob
import os

files = sorted(glob.glob("ercot_rtm_*.parquet"))
results = []

for f in files:
    year = f.replace("ercot_rtm_", "").replace(".parquet", "")
    df = pd.read_parquet(f)
    
    # Filter for HB_NORTH
    north = df[df['Location'] == 'HB_NORTH'].copy()
    
    if not north.empty:
        # Convert Time_Central to datetime if it's not already
        north['Time_Central'] = pd.to_datetime(north['Time_Central'])
        
        # Filter for Jan 1 to May 16 (inclusive)
        ytd_mask = (north['Time_Central'].dt.month < 5) | ((north['Time_Central'].dt.month == 5) & (north['Time_Central'].dt.day <= 16))
        north_ytd = north[ytd_mask]
        
        if not north_ytd.empty:
            avg_price = north_ytd['SPP'].mean()
            results.append({'Year': year, 'YTD Average Price ($/MWh)': round(avg_price, 2)})
        else:
            results.append({'Year': year, 'YTD Average Price ($/MWh)': 'No Data in Window'})
    else:
        results.append({'Year': year, 'YTD Average Price ($/MWh)': 'N/A'})

result_df = pd.DataFrame(results)
print(result_df.to_markdown(index=False))
