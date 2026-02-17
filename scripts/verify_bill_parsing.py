import pandas as pd
import os

FILE_PATH = "AzureSkyActuals.xlsx"

def verify_parsing():
    if not os.path.exists(FILE_PATH):
        print(f"File {FILE_PATH} not found.")
        return

    print(f"Reading {FILE_PATH}...")
    df = pd.read_excel(FILE_PATH)
    print("Columns:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head())
    print("\nTypes:")
    print(df.dtypes)
    
    # Check Date Column
    if 'Date' in df.columns:
        print("\nDate Column Sample:")
        print(df['Date'].head())
        # Check if it's already datetime
        is_dt = pd.api.types.is_datetime64_any_dtype(df['Date'])
        print(f"Is Date datetime? {is_dt}")
        
    # Check Generation Column
    gen_col = "Plant Generation (MWh)"
    if gen_col in df.columns:
        print(f"\n{gen_col} Sample:")
        print(df[gen_col].head())
        print("Total Gen:", df[gen_col].sum())
    else:
        print(f"\nColumn '{gen_col}' not found!")

if __name__ == "__main__":
    verify_parsing()
