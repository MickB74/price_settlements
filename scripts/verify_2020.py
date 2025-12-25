import gridstatus
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import patch_gridstatus
import numpy as np

def test_processing():
    iso = gridstatus.Ercot()
    try:
        print("Fetching 2020 data...")
        # Fetch a small chunk if possible, but we know it fetches the year.
        # We'll just fetch it again (cached by gridstatus internally usually, or we just wait a bit)
        # To save time, if we could load the file we just downloaded... gridstatus saves to tmp usually.
        # But let's just call it, it should be fast enough or cached.
        df = iso.get_rtm_spp(year=2020)
        print(f"Fetched {len(df)} rows.")
        
        # Mimic app.py processing
        print("Processing data...")
        if not pd.api.types.is_datetime64_any_dtype(df['Time']):
            df['Time'] = pd.to_datetime(df['Time'], utc=True)
        
        df['Time_Central'] = df['Time'].dt.tz_convert('US/Central')
        print("Time conversion successful.")
        print(df[['Time', 'Time_Central']].head())
        
        # Check for expected columns
        expected_cols = ['Location', 'SPP', 'Time_Central']
        for col in expected_cols:
            if col not in df.columns:
                print(f"MISSING COLUMN: {col}")
            else:
                print(f"Found column: {col}")
                
        print("Verification successful!")
        
    except Exception as e:
        print(f"Error in processing: {e}")

if __name__ == "__main__":
    test_processing()
