import pandas as pd
import glob
import os

files = sorted(glob.glob("ercot_rtm_*.parquet"))
results = []

for f in files:
    year = f.replace("ercot_rtm_", "").replace(".parquet", "")
    df = pd.read_parquet(f)
    
    # Filter for HB_NORTH
    north = df[df['Location'] == 'HB_NORTH']
    if not north.empty:
        avg_price = north['SPP'].mean()
        results.append({'Year': year, 'Average Price ($/MWh)': round(avg_price, 2)})
    else:
        results.append({'Year': year, 'Average Price ($/MWh)': 'N/A'})

result_df = pd.DataFrame(results)
print(result_df.to_markdown(index=False))
