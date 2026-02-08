import pandas as pd
import glob
import os
import sys

# Check pregenerated profiles
files = glob.glob("data_cache/pregenerated/*.parquet")

if not files:
    print("No pregenerated files found in data_cache/pregenerated/")
    sys.exit(1)

print(f"Checking {len(files)} profiles...")

results = []
for f in files:
    try:
        df = pd.read_parquet(f)
        name = os.path.basename(f)
        
        if df.empty or 'gen_mw' not in df.columns:
            continue
            
        # Standard capacity was 100 MW in pregeneration script for all profiles
        capacity = 100.0 
        
        mean_mw = df['gen_mw'].mean()
        cf = mean_mw / capacity
        
        # Extract metadata
        # Filename format: HUB_NAME_Tech_Year.parquet
        # e.g. HB_NORTH_Solar_2005.parquet
        base = name.replace('.parquet', '')
        parts = base.split('_')
        # Parts: ['HB', 'NORTH', 'Solar', '2005'] -> Hub is "HB_NORTH"
        # Parts: ['HB', 'HOUSTON', 'Solar', '2005'] -> Hub is "HB_HOUSTON"
        
        year = int(parts[-1])
        tech = parts[-2]
        hub = "_".join(parts[:-2])
        
        results.append({
            'hub': hub,
            'tech': tech,
            'year': year,
            'cf': cf
        })
    except Exception as e:
        print(f"Error processing {f}: {e}")

if not results:
    print("No valid results.")
    sys.exit(1)

df_res = pd.DataFrame(results)

print("\n" + "="*60)
print("CAPACITY FACTOR Averages (%)")
print("="*60)

print("\n--- Solar ---")
solar_cfs = df_res[df_res['tech']=='Solar'].groupby('hub')['cf'].mean() * 100
print(solar_cfs)

print("\n--- Wind ---")
wind_cfs = df_res[df_res['tech']=='Wind'].groupby('hub')['cf'].mean() * 100
print(wind_cfs)

print("\nDetailed Stats:")
print(df_res.groupby(['hub', 'tech'])['cf'].describe())
