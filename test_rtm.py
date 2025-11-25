import gridstatus
from datetime import datetime
import inspect

def test_rtm_2024():
    try:
        iso = gridstatus.Ercot()
        print("Inspecting get_rtm_spp signature:")
        if hasattr(iso, 'get_rtm_spp'):
            print(inspect.signature(iso.get_rtm_spp))
            
            print("\nListing all 'get_' methods on iso object:")
            print([method for method in dir(iso) if method.startswith('get_')])

            print(f"\nFetching ERCOT Real-Time prices for year 2024...")
            
            # Try with year argument
            try:
                # We'll just fetch the head to avoid printing too much if it works
                df = iso.get_rtm_spp(year=2024)
                print("Successfully fetched RTM SPP using year=2024")
                print(df.head())
                print(df.columns)
                print(f"Row count: {len(df)}")
                
                # Check if we can filter for a specific day easily
                target_date = "2024-01-01"
                print(f"\nFiltering for {target_date}...")
                # Assuming 'Time' or 'Interval Ending' column exists and is datetime
                # We need to see columns first, but let's try a generic filter if possible or just print columns
                
            except Exception as e:
                print(f"Failed with year=2024: {e}")
                
        else:
            print("get_rtm_spp method not found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_rtm_2024()
