import pandas as pd
import os

INPUT_FILE = "AzureSkyActuals.xlsx"
OUTPUT_FILE = "sced_cache/Settlement_Invoice_Actuals.parquet"

def convert_bill():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except Exception as e:
        print(f"Failed to read Excel: {e}")
        return

    # Check columns
    req_cols = ['Date', 'Plant Generation (MWh)']
    if not all(col in df.columns for col in req_cols):
        print(f"Missing columns. Found: {df.columns.tolist()}")
        return

    # Standardize
    print("Processing data...")
    # Sort by Date and Settlement Interval to ensure correct order for ambiguous time inference
    if 'Settlement Interval' in df.columns:
        df = df.sort_values(['Date', 'Settlement Interval']).reset_index(drop=True)
    else:
        df = df.sort_values('Date').reset_index(drop=True)

    df['Time'] = pd.to_datetime(df['Date'])
    
    # Localize to Central
    # ambiguous='infer' relies on the order being correct (falling back)
    if df['Time'].dt.tz is None:
        try:
            df['Time'] = df['Time'].dt.tz_localize('US/Central', ambiguous='infer', nonexistent='shift_forward')
        except Exception as e:
            print(f"Standard localization failed: {e}. Trying fallback...")
            # Fallback: assume sorted, use position? 
            # Or use ambiguous=[True/False] based on interval?
            # If 'Settlement Interval' exists:
            # On DST end day, intervals 5-8 are CDT, 9-12 are CST? 
            # Actually, standard is 02:00 fallback.
            # 01:00-02:00 happens twice.
            # First one is CDT (offset -0500), second is CST (offset -0600).
            # 'infer' should work if sorted.
            # If it fails, let's try a safe approach dropping ambiguous for now or using generic UTC.
            # But specific logic is better.
            # Let's try forcing the first occurrence to be DST=True and second to False?
            # Too complex for this script?
            # Let's try to just use 'NaT' and warn.
            print("Warning: Ambiguous times found. Dropping them to proceed.")
            df['Time'] = df['Time'].dt.tz_localize('US/Central', ambiguous='NaT', nonexistent='shift_forward')
            df = df.dropna(subset=['Time'])
    else:
        df['Time'] = df['Time'].dt.tz_convert('US/Central')
    
    # Calculate MW from MWh (15-min intervals)
    # MW = MWh * 4
    df['Actual_MW'] = pd.to_numeric(df['Plant Generation (MWh)'], errors='coerce') * 4.0
    
    # Extract Price if available (Floating Price)
    if 'Floating Price (RT_ERCOT HB_NORTH)' in df.columns:
        df['Settlement_Point_Price'] = pd.to_numeric(df['Floating Price (RT_ERCOT HB_NORTH)'], errors='coerce')
    else:
        # Try generic
        price_col = next((c for c in df.columns if 'Floating Price' in c), None)
        if price_col:
            df['Settlement_Point_Price'] = pd.to_numeric(df[price_col], errors='coerce')
        else:
            df['Settlement_Point_Price'] = 0.0

    # Ensure clean numeric
    df = df.dropna(subset=['Time'])
    df['Actual_MW'] = df['Actual_MW'].fillna(0.0)
    
    # Select final columns matching typical SCED/RTM structure
    final_df = df[['Time', 'Actual_MW', 'Settlement_Point_Price']].copy()
    
    # Sort
    final_df = final_df.sort_values('Time').reset_index(drop=True)
    
    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_parquet(OUTPUT_FILE)
    print(f"Done! Saved {len(final_df)} rows. Range: {final_df['Time'].min()} to {final_df['Time'].max()}")

if __name__ == "__main__":
    convert_bill()
