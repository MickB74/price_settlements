import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILES = [
    REPO_ROOT / "AzureSkyActuals.xlsx",
    REPO_ROOT / "AzureSkyActuals2.xlsx",
    REPO_ROOT / "AzureSkyActualsJan2026.xlsx",
]
OUTPUT_FILE = REPO_ROOT / "data_static" / "Settlement_Invoice_Actuals.parquet"
LEGACY_OUTPUT_FILE = REPO_ROOT / "sced_cache" / "Settlement_Invoice_Actuals.parquet"

def _find_header_row(input_file: Path) -> int:
    """Scan the first 20 rows to find which row contains the 'Date' column header."""
    try:
        raw = pd.read_excel(input_file, header=None, nrows=20)
    except Exception:
        return 0
    for i, row in raw.iterrows():
        if any(str(v).strip() == 'Date' for v in row):
            return i
    return 0  # Default: first row is the header

def _convert_single_bill(input_file: Path):
    print(f"Reading {input_file}...")
    header_row = _find_header_row(input_file)
    if header_row > 0:
        print(f"  Detected header at row {header_row} in {input_file.name}")
    try:
        df = pd.read_excel(input_file, header=header_row)
    except Exception as e:
        print(f"Failed to read Excel {input_file}: {e}")
        return None

    # Check columns
    req_cols = ['Date', 'Plant Generation (MWh)']
    if not all(col in df.columns for col in req_cols):
        print(f"Missing columns in {input_file}. Found: {df.columns.tolist()}")
        return None

    # Standardize
    print(f"Processing data from {input_file.name}...")
    # Sort by Date and Settlement Interval to ensure correct order for ambiguous time inference
    if 'Settlement Interval' in df.columns:
        df = df.sort_values(['Date', 'Settlement Interval']).reset_index(drop=True)
    else:
        df = df.sort_values('Date').reset_index(drop=True)

    # Drop non-data rows (e.g. "Totals:" footer rows in newer invoice formats)
    df = df[pd.to_datetime(df['Date'], errors='coerce').notna()].copy()

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
    return final_df

def convert_bill():
    available_inputs = [p for p in INPUT_FILES if p.exists()]
    if not available_inputs:
        print(f"Error: none of the input files exist: {', '.join(str(p) for p in INPUT_FILES)}")
        return

    converted = []
    for path in available_inputs:
        part = _convert_single_bill(path)
        if part is not None and not part.empty:
            converted.append((path, part))

    if not converted:
        print("Error: no valid input data to convert.")
        return

    # Use AzureSkyActuals.xlsx as the base if present, then append new timestamps from others.
    merged = converted[0][1]
    if len(converted) > 1:
        base_times = set(merged["Time"])
        for path, part in converted[1:]:
            add = part[~part["Time"].isin(base_times)].copy()
            # Defensive cleanup for repeated placeholder rows in supplemental files.
            add = add.drop_duplicates(subset=["Time"], keep="first")
            base_times.update(add["Time"])
            merged = pd.concat([merged, add], ignore_index=True)
            print(f"Merged {len(add)} unique rows from {path.name}.")

    final_df = merged.sort_values("Time").reset_index(drop=True)

    # Save canonical cloud-safe output path.
    print(f"Saving to {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUTPUT_FILE)

    # Also write legacy path for local compatibility.
    LEGACY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(LEGACY_OUTPUT_FILE)
    print(f"Done! Saved {len(final_df)} rows. Range: {final_df['Time'].min()} to {final_df['Time'].max()}")

if __name__ == "__main__":
    convert_bill()
