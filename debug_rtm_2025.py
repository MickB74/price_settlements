import gridstatus
import traceback

try:
    print("Initializing ERCOT...")
    iso = gridstatus.Ercot()
    print("Fetching RTM SPP for 2025...")
    df = iso.get_rtm_spp(year=2025)
    print("Success!")
    print(df.head())
except Exception:
    traceback.print_exc()
