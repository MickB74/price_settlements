import gridstatus
import pandas as pd
import os
import pytest

def test_ercot_range():
    if os.getenv("RUN_LIVE_ERCOT_TESTS") != "1":
        pytest.skip("Live ERCOT API test skipped. Set RUN_LIVE_ERCOT_TESTS=1 to enable.")

    iso = gridstatus.Ercot()
    year = 2024
    print(f"Fetching RTM SPP for {year}...")
    df = iso.get_rtm_spp(year=year)
    assert not df.empty
    assert "Time" in df.columns
    
    df['Time_UTC'] = pd.to_datetime(df['Time'], utc=True)
    df['Time_CST'] = df['Time_UTC'].dt.tz_convert('US/Central')
    
    print(f"Min UTC: {df['Time_UTC'].min()}")
    print(f"Max UTC: {df['Time_UTC'].max()}")
    print(f"Max CST: {df['Time_CST'].max()}")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    test_ercot_range()
