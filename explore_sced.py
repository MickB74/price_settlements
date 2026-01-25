import gridstatus
import pandas as pd
from datetime import datetime, timedelta

def explore_sced():
    iso = gridstatus.Ercot()
    # 60 days ago
    target_date = datetime.now() - timedelta(days=65)
    print(f"Fetching 60-day SCED disclosure for {target_date.date()}...")
    
    try:
        data = iso.get_60_day_sced_disclosure(date=target_date)
        if isinstance(data, dict):
            print(f"Returned a dictionary with keys: {list(data.keys())}")
            for key, df in data.items():
                print(f"\n--- Key: {key} ---")
                if isinstance(df, pd.DataFrame):
                    print(f"Rows: {len(df)}")
                    print(f"Columns: {df.columns.tolist()}")
                    print(df.head(2))
                    
                    # Search for Resource names
                    if 'Resource Name' in df.columns or 'Unit Name' in df.columns:
                        res_col = 'Resource Name' if 'Resource Name' in df.columns else 'Unit Name'
                        print(f"Sample Resources in {key}:")
                        print(df[res_col].unique()[:5])
        else:
            print("Returned a single DataFrame.")
            print(data.head())
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    explore_sced()
