import fetch_tmy
import pandas as pd

HUB_LOCATIONS = {
    "HB_NORTH": (32.3865, -96.8475),
    "HB_SOUTH": (26.9070, -99.2715),
    "HB_WEST": (32.4518, -100.5371),
    "HB_HOUSTON": (29.3013, -94.7977),
    "HB_PAN": (35.2220, -101.8313),
}

print("Comparing 2024 Actual vs TMY Wind Generation\n")
print(f"{'Hub':<15} {'2024 (GWh)':<15} {'TMY (GWh)':<15} {'Diff %':<10}")
print("=" * 60)

for hub_name, (lat, lon) in HUB_LOCATIONS.items():
    # Get 2024 actual
    profile_2024 = fetch_tmy.get_profile_for_year(
        year=2024, tech="Wind", capacity_mw=80, 
        lat=lat, lon=lon, force_tmy=False
    )
    gen_2024 = profile_2024.sum() * 0.25 / 1000  # Convert to GWh
    
    # Get TMY
    profile_tmy = fetch_tmy.get_profile_for_year(
        year=2024, tech="Wind", capacity_mw=80,
        lat=lat, lon=lon, force_tmy=True
    )
    gen_tmy = profile_tmy.sum() * 0.25 / 1000  # Convert to GWh
    
    diff_pct = ((gen_2024 - gen_tmy) / gen_tmy * 100) if gen_tmy > 0 else 0
    
    print(f"{hub_name:<15} {gen_2024:<15.1f} {gen_tmy:<15.1f} {diff_pct:>+7.1f}%")
