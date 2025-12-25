import gridstatus
from datetime import datetime, timedelta

def list_ercot_hubs():
    try:
        iso = gridstatus.Ercot()
        today = datetime.now().date()
        
        print(f"Testing DAY_AHEAD_HOURLY for {today} WITHOUT location filter...")
        try:
            # Remove location_type="Hub" to get everything
            df = iso.get_spp(
                date=today,
                market="DAY_AHEAD_HOURLY"
            )
            
            if not df.empty:
                print(f"\nSUCCESS: Found data for {today}")
                print("Columns:", df.columns.tolist())
                print("First 5 rows:")
                print(df.head().to_string())
                
                if "Location" in df.columns:
                    # Check if HB_NORTH is in there
                    if "HB_NORTH" in df["Location"].values:
                        print("\nHB_NORTH found!")
                        # Print the row for HB_NORTH to see its Location Type
                        print(df[df["Location"] == "HB_NORTH"].head().to_string())
                    else:
                        print("\nHB_NORTH NOT found in Location column.")
            else:
                print(f"No data for {today} even without filter.")
        except Exception as e:
            print(f"Error fetching: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    list_ercot_hubs()
