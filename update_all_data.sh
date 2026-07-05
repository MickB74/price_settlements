#!/bin/bash

# Exit on any error
set -e

echo "========================================="
echo "⚡️ Starting ERCOT Data Update Pipeline ⚡️"
echo "========================================="

# 1. Update ERCOT Real-Time Prices
echo ""
echo "[1/4] Updating ERCOT RTM Prices..."
.venv/bin/python update_ercot_2025.py
.venv/bin/python update_ercot_2026.py

# 2. Pre-generate actual weather profiles for Hubs
echo ""
echo "[2/4] Pre-generating Solar & Wind Profiles..."
.venv/bin/python pregenerate_2025_profiles.py
.venv/bin/python pregenerate_2026_profiles.py

# 3. Update SCED Full Disclosures (60-day lag)
echo ""
echo "[3/4] Updating SCED Full Disclosures..."
.venv/bin/python update_full_disclosure.py

# 4. Update Azure Sky Wind Aggregation (from SCED)
echo ""
echo "[4/4] Updating Azure Sky Wind Aggregation..."
.venv/bin/python download_azure_wind.py

echo ""
echo "========================================="
echo "✅ All data successfully updated!"
echo "Next step: Restart the Streamlit app to clear the memory cache."
echo "========================================="
