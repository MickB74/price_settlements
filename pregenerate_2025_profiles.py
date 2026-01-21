#!/usr/bin/env python3
"""
Pre-generate 2025 actual weather profiles for all ERCOT hubs.
This ensures the cache is populated with actual 2025 data before running the app.
"""

import fetch_tmy
import importlib

# Force reload to get latest code
importlib.reload(fetch_tmy)

# Hub locations from app.py
HUB_LOCATIONS = {
    "HB_NORTH": (32.3865, -96.8475),   # Waxahachie, TX
    "HB_SOUTH": (26.9070, -99.2715),   # Zapata, TX
    "HB_WEST": (32.4518, -100.5371),   # Roscoe, TX
    "HB_HOUSTON": (29.3013, -94.7977), # Galveston, TX
    "HB_PAN": (35.2220, -101.8313),    # Amarillo, TX
}

print("=" * 70)
print("Pre-generating 2025 Actual Weather Profiles for All Hubs")
print("=" * 70)

capacity_mw = 100  # Standard test capacity

for hub_name, (lat, lon) in HUB_LOCATIONS.items():
    print(f"\n{hub_name} ({lat}, {lon})")
    print("-" * 70)
    
    # Generate Solar Profile
    print(f"  Generating 2025 Solar profile...")
    solar_profile = fetch_tmy.get_profile_for_year(
        year=2025,
        tech="Solar",
        capacity_mw=capacity_mw,
        lat=lat,
        lon=lon,
        force_tmy=False  # Use actual data
    )
    
    if not solar_profile.empty:
        print(f"    ✓ Solar: {len(solar_profile):,} points, Mean: {solar_profile.mean():.2f} MW")
    else:
        print(f"    ✗ Solar: Failed to generate")
    
    # Generate Wind Profile
    print(f"  Generating 2025 Wind profile...")
    wind_profile = fetch_tmy.get_profile_for_year(
        year=2025,
        tech="Wind",
        capacity_mw=capacity_mw,
        lat=lat,
        lon=lon,
        force_tmy=False  # Use actual data
    )
    
    if not wind_profile.empty:
        print(f"    ✓ Wind: {len(wind_profile):,} points, Mean: {wind_profile.mean():.2f} MW")
    else:
        print(f"    ✗ Wind: Failed to generate")

print("\n" + "=" * 70)
print("✅ All 2025 actual weather profiles pre-generated!")
print("=" * 70)
print("\nNext step: Restart your Streamlit app to use the new data")
print("  streamlit run app.py")
