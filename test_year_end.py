import pandas as pd
import fetch_tmy

def test_year_end():
    year = 2024
    tech = "Solar"
    capacity = 100
    lat, lon = 32.4487, -99.7331
    
    print(f"Generating profile for {year}...")
    profile = fetch_tmy.get_profile_for_year(year, tech, capacity, lat, lon)
    
    if profile.empty:
        print("Profile is empty!")
        return
        
    print(f"Profile range: {profile.index.min()} to {profile.index.max()}")
    
    # Check last day CST
    pc = profile.tz_convert('US/Central')
    last_day = pc[pc.index.date == pd.Timestamp(f"{year}-12-31").date()]
    print(f"\nLast day Central Time ({year}-12-31) intervals:")
    print(last_day.tail(20))
    
    missing_count = 96 - len(last_day)
    if missing_count > 0:
        print(f"\n⚠️ WARNING: Missing {missing_count} intervals for the last day in Central Time.")
    else:
        print("\n✅ All 96 intervals for the last day are present in Central Time.")

if __name__ == "__main__":
    test_year_end()
