import os
import pandas as pd
import glob
from datetime import datetime

# Parameters
CACHE_DIR = "sced_cache"
RESOURCES = ["VORTEX_WIND1", "VORTEX_WIND2", "VORTEX_WIND3", "VORTEX_WIND4"]
AGG_NAME = "AZURE_SKY_WIND_AGG"
START_DATE_LIMIT = datetime(2024, 10, 1).date()

def main():
    print(f"Aggregating {AGG_NAME} from local cache...")
    
    # List all daily disclosure files
    pattern = os.path.join(CACHE_DIR, "full_disclosure_*.parquet")
    files = glob.glob(pattern)
    print(f"Found {len(files)} cached daily files.")
    
    all_data = []
    
    for f in files:
        # Extract date from filename: full_disclosure_YYYY-MM-DD.parquet
        basename = os.path.basename(f)
        date_str = basename.replace("full_disclosure_", "").replace(".parquet", "")
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
            
        if date_obj < START_DATE_LIMIT:
            continue
            
        try:
            df = pd.read_parquet(f)
            
            # Filter for VORTEX resources
            # Normalize column names just in case
            df.columns = [c.strip() for c in df.columns]
            
            if 'Resource Name' not in df.columns:
                continue
                
            mask = df['Resource Name'].isin(RESOURCES)
            df_vortex = df[mask].copy()
            
            if df_vortex.empty:
                continue
                
            # Process timestamps
            if 'Interval Start' in df_vortex.columns:
                df_vortex['Time'] = pd.to_datetime(df_vortex['Interval Start'], utc=True)
            elif 'SCED Timestamp' in df_vortex.columns:
                df_vortex['Time'] = pd.to_datetime(df_vortex['SCED Timestamp'], utc=True).dt.floor('15min') # Approximate
                
            # Resample/Sum logic
            # We want to sum all VORTEX units for each timestamp
            # But raw SCED data is irregular (~every 5 mins).
            # We need to resample to 15-min.
            
            # First, simply keep relevant cols
            cols = ['Time', 'Resource Name', 'Telemetered Net Output']
            if 'Telemetered Net Output' not in df_vortex.columns:
                continue
                
            df_subset = df_vortex[cols].dropna()
            
            # Resample each resource to 15-min mean FIRST
            # Then sum across resources
            
            resampled_dfs = []
            for res in RESOURCES:
                curr = df_subset[df_subset['Resource Name'] == res]
                if curr.empty:
                    continue
                
                # Resample to 15min mean
                # Note: SCED is 15-min settlement but reported frequently. 
                # Settlement uses time-weighted average? Or simple mean? 
                # sced_fetcher uses .resample('15min').mean()
                
                r = curr.set_index('Time')[['Telemetered Net Output']].resample('15min').mean()
                r.columns = [res]
                resampled_dfs.append(r)
                
            if not resampled_dfs:
                continue
                
            # Join all units for this day
            day_panel = pd.concat(resampled_dfs, axis=1)
            
            # Sum rows (Actual_MW)
            day_panel['Actual_MW'] = day_panel.sum(axis=1) # Sum available units
            
            # Store result
            result = day_panel[['Actual_MW']].reset_index()
            all_data.append(result)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if not all_data:
        print("No valid data found in cache.")
        return

    print(f"Combiniing {len(all_data)} daily chunks...")
    full_df = pd.concat(all_data).sort_values('Time').drop_duplicates('Time')
    
    # Save by year
    years = full_df['Time'].dt.year.unique()
    for year in years:
        df_year = full_df[full_df['Time'].dt.year == year].copy()
        out_path = os.path.join(CACHE_DIR, f"{AGG_NAME}_{year}_full.parquet")
        df_year.to_parquet(out_path)
        print(f"Saved {year} data to {out_path} ({len(df_year)} rows)")

if __name__ == "__main__":
    main()
