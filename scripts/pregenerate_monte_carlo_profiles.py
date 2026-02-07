#!/usr/bin/env python3
"""
Pregenerate all weather profiles needed for Monte Carlo simulation.

This script generates profiles for all combinations of:
- Years: 2005-2024 (20 years for Monte Carlo sampling)
- Technologies: Solar, Wind
- Hub Locations: HB_NORTH, HB_SOUTH, HB_WEST, HB_HOUSTON, HB_PAN

The generated profiles are saved to data_cache/pregenerated/ and should be
committed to the repo to avoid API dependency issues on Streamlit Cloud.
"""

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fetch_tmy

# Hub locations from app.py
HUB_LOCATIONS = {
    "HB_NORTH": (32.3865, -96.8475),
    "HB_SOUTH": (26.9070, -99.2715),
    "HB_WEST": (32.4518, -100.5371),
    "HB_HOUSTON": (29.3013, -94.7977),
    "HB_PAN": (35.2220, -101.8313),
}

YEARS = list(range(2005, 2025))  # 2005-2024
TECHS = ["Solar", "Wind"]
CAPACITY_MW = 100  # Normalized to 100 MW (profiles scale linearly)

PREGEN_DIR = "data_cache/pregenerated"
os.makedirs(PREGEN_DIR, exist_ok=True)

def main():
    total = len(HUB_LOCATIONS) * len(TECHS) * len(YEARS)
    count = 0
    failures = []
    
    print(f"Generating {total} profiles...")
    print(f"Output directory: {PREGEN_DIR}/")
    print()
    
    for hub_name, (lat, lon) in HUB_LOCATIONS.items():
        for tech in TECHS:
            for year in YEARS:
                count += 1
                filename = f"{hub_name}_{tech}_{year}.parquet"
                filepath = os.path.join(PREGEN_DIR, filename)
                
                # Skip if already exists and valid
                if os.path.exists(filepath):
                    try:
                        df = pd.read_parquet(filepath)
                        if not df.empty and len(df) > 30000:  # Should have ~35040 rows
                            print(f"[{count}/{total}] ✓ Skipping {filename} (already exists)")
                            continue
                    except:
                        pass
                
                print(f"[{count}/{total}] Generating {filename}...")
                
                try:
                    # Generate profile
                    profile = fetch_tmy.get_profile_for_year(
                        year=year,
                        tech=tech,
                        capacity_mw=CAPACITY_MW,
                        lat=lat,
                        lon=lon,
                        force_tmy=False,
                        turbine_type="GENERIC" if tech == "Wind" else None,
                        tracking=True if tech == "Solar" else None,
                        efficiency=0.86
                    )
                    
                    if profile is None or len(profile) == 0:
                        failures.append(f"{filename}: Empty profile returned")
                        print(f"   ❌ FAILED: Empty profile")
                        continue
                    
                    # Save to parquet
                    df = pd.DataFrame({
                        'datetime': profile.index,
                        'gen_mw': profile.values
                    })
                    df.to_parquet(filepath, index=False)
                    print(f"   ✓ Saved ({len(profile)} intervals)")
                    
                except Exception as e:
                    failures.append(f"{filename}: {str(e)}")
                    print(f"   ❌ ERROR: {e}")
    
    print()
    print("=" * 60)
    print(f"Generation complete: {count - len(failures)}/{total} succeeded")
    
    if failures:
        print(f"\n{len(failures)} failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    else:
        print("\n✅ All profiles generated successfully!")
        print(f"\nNext steps:")
        print(f"1. git add {PREGEN_DIR}")
        print(f"2. git commit -m 'Add pregenerated Monte Carlo profiles'")
        print(f"3. git push")
        return 0

if __name__ == "__main__":
    sys.exit(main())
