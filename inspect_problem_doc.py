import pandas as pd
import requests
from io import BytesIO
import zipfile

url = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId=1191921328"
print(f"Downloading {url}...")
r = requests.get(url)
print(f"Status: {r.status_code}")

try:
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        print("Files in zip:")
        print(z.namelist())
        for fname in z.namelist():
            if fname.endswith('csv') or fname.endswith('xml'):
                print(f"\n--- Content of {fname} head ---")
                with z.open(fname) as f:
                    # Read first few lines
                    head = [f.readline().decode('utf-8') for _ in range(5)]
                    for line in head:
                        print(line.strip())
                    
                    # Try pandas read
                    f.seek(0)
                    try:
                        df = pd.read_csv(f)
                        print("\nPandas parsed columns:")
                        print(list(df.columns))
                    except Exception as e:
                        print(f"\nPandas read failed: {e}")

except Exception as e:
    print(f"Zip error: {e}")
    # maybe it's not a zip?
    print("Raw content head:")
    print(r.content[:200])
