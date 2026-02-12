import gridstatus
import pandas as pd

def find_resource():
    iso = gridstatus.Ercot()
    target_date = "2024-11-01"
    print(f"Fetching SCED disclosure for {target_date}...")
    
    try:
        data = iso.get_60_day_sced_disclosure(date=target_date)
        if 'sced_gen_resource' in data:
            df = data['sced_gen_resource']
            
            # Candidates
            candidates = ["SEVEN", "COWBOY", "ROAD", "RUNNER", "RATTLE", "SNAKE", "WHISPER", "ASPEN"]
            
            print("\n--- Other Enel Candidates ---")
            for cand in candidates:
                matches = df[df['Resource Name'].str.contains(cand, na=False)]
                if not matches.empty:
                    print(f"\nMatches for '{cand}':")
                    cols = ['Resource Name', 'QSE']
                    print(matches[cols].drop_duplicates().to_string(index=False))

        else:
            print("No 'sced_gen_resource' table in data.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_resource()
