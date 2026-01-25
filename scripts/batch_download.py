import sys
import os
import subprocess
import time

ASSETS = [
    # North Solar
    "ROSELAND_SOLAR1",
    "JKLP_SLR_PV1",
    # West Solar
    "ASCK_SLR_SOLAR1",
    # West Wind
    "AJAXWIND_UNIT1",
    "MROW_SLR_SOLAR1",
    # South Wind
    "MONTECR1_WIND1"
]

YEARS = [2024, 2025]

def main():
    print(f"Starting batch download for {len(ASSETS)} assets over {len(YEARS)} years...")
    print(f"Total jobs: {len(ASSETS) * len(YEARS)}")
    
    # We will launch them sequentially (blocking) to avoid overwhelming the system/API limits
    # But each job is internally multi-threaded
    
    for asset in ASSETS:
        for year in YEARS:
            print(f"\n>>> Launching: {asset} ({year})")
            cmd = [
                sys.executable, 
                "scripts/bulk_fetch_sced.py", 
                asset, 
                "--year", str(year),
                "--workers", "6" 
            ]
            
            # Using subprocess.run to wait for completion before starting next
            # This ensures we don't open 500 threads at once
            try:
                subprocess.run(cmd, check=True)
                print(f">>> Completed: {asset} ({year})")
            except subprocess.CalledProcessError as e:
                print(f"!!! Failed: {asset} ({year}) - {e}")
            
            # Small cooldown
            time.sleep(2)

if __name__ == "__main__":
    main()
