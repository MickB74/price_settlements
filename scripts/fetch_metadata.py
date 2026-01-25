import requests
import os
import shutil
import zipfile
import io

# Direct Zip Link from USGS
USWTDB_ZIP_URL = "https://energy.usgs.gov/uswtdb/assets/data/uswtdbCSV.zip"
STATIC_DIR = "data_static"
OUTPUT_FILE = os.path.join(STATIC_DIR, "uswtdb.csv")

def download_metadata():
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR)
        
    print(f"Downloading US Wind Turbine Database from {USWTDB_ZIP_URL}...")
    try:
        with requests.get(USWTDB_ZIP_URL, stream=True) as r:
            r.raise_for_status()
            
            # Unzip in memory
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                # Find the main CSV file (usually uswtdb_vX_X.csv)
                csv_file = next((n for n in z.namelist() if n.endswith('.csv') and 'uswtdb' in n), None)
                
                if csv_file:
                    print(f"Extraction target: {csv_file}")
                    with z.open(csv_file) as source, open(OUTPUT_FILE, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    print(f"✅ Successfully downloaded and extracted to {OUTPUT_FILE}")
                else:
                    print("❌ No valid CSV found in zip archive.")

    except Exception as e:
        print(f"❌ Failed to download USWTDB: {e}")

if __name__ == "__main__":
    download_metadata()
