import gridstatus
import pandas as pd
import inspect

def check_fuel_mix():
    try:
        iso = gridstatus.Ercot()
        print("Listing 'get_' methods on iso object:")
        print([method for method in dir(iso) if method.startswith('get_')])

        print("\nTesting get_fuel_mix with start/end...")
        start_date = "2024-01-01"
        end_date = "2024-01-02"
        
        try:
            if hasattr(iso, 'get_fuel_mix'):
                # Some gridstatus methods accept start/end even if signature says date
                try:
                    df = iso.get_fuel_mix(start=start_date, end=end_date)
                    print("Success with start/end!")
                    print(df.head())
                except TypeError:
                    print("start/end arguments not accepted.")
                    # Fallback: try loop for 2 days
                    print("Trying loop for 2 days...")
                    dfs = []
                    for d in ["2024-01-01", "2024-01-02"]:
                        try:
                            print(f"Fetching {d}...")
                            dfs.append(iso.get_fuel_mix(date=d))
                        except Exception as e:
                            print(f"Failed for {d}: {e}")
                    
                    if dfs:
                        df = pd.concat(dfs)
                        print(df.head())
            else:
                print("Method not found.")
        except Exception as e:
            print(f"Error fetching fuel mix: {e}")

    except Exception as e:
        print(f"Error initializing ISO: {e}")

if __name__ == "__main__":
    check_fuel_mix()
