import gridstatus
import pandas as pd

def test_ercot_range():
    iso = gridstatus.Ercot()
    year = 2024
    print(f"Fetching RTM SPP for {year}...")
    df = iso.get_rtm_spp(year=year)
    
    df['Time_UTC'] = pd.to_datetime(df['Time'], utc=True)
    df['Time_CST'] = df['Time_UTC'].dt.tz_convert('US/Central')
    
    print(f"Min UTC: {df['Time_UTC'].min()}")
    print(f"Max UTC: {df['Time_UTC'].max()}")
    print(f"Max CST: {df['Time_CST'].max()}")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    test_ercot_range()
