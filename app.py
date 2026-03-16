import streamlit as st
import os
import sys
import subprocess
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import gridstatus
import patch_gridstatus # Apply monkey patch
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from utils import power_curves, variability_analysis, monte_carlo
from datetime import datetime, timedelta
import time
import zipfile
import io
import fetch_tmy # New module for TMY data
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import tempfile
import folium
from utils.pdf_generator import generate_settlement_pdf
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import sced_fetcher
import json
import tabs.azure_comparison as tab_azure
import tabs.vppa_8760_compare as tab_vppa_8760_compare

try:
    from utils.wind_calibration import apply_congestion_haircut, get_offline_threshold_mw
except Exception:
    # Keep app startup resilient if wind_calibration import breaks in a partial deploy.
    def get_offline_threshold_mw(capacity_mw=None, pct_of_capacity=0.05, min_mw=2.0, max_mw=20.0):
        try:
            cap = float(capacity_mw)
        except (TypeError, ValueError):
            cap = None
        if cap is None or cap <= 0:
            return 5.0
        threshold = cap * pct_of_capacity
        return float(max(min_mw, min(max_mw, threshold)))

    def apply_congestion_haircut(gen_series, spp_series, hub_name=None, resource_id=None, calibration_table=None):
        return gen_series

# --- Constants & Configuration ---
REPO_ROOT = Path(__file__).resolve().parent
SETTLEMENT_INVOICE_XLSX = REPO_ROOT / "AzureSkyActuals.xlsx"
SETTLEMENT_INVOICE_PARQUET = REPO_ROOT / "data_static" / "Settlement_Invoice_Actuals.parquet"
SETTLEMENT_INVOICE_PARQUET_LEGACY = REPO_ROOT / "sced_cache" / "Settlement_Invoice_Actuals.parquet"
# Invoice floating-price feed is offset by one 15-min interval vs SCED timeline.
# Shift back one row (negative shift) so each timestamp uses the current interval price.
INVOICE_PRICE_ALIGNMENT_SHIFT_INTERVALS = -1

HUB_LOCATIONS = {
    "HB_NORTH": (32.3865, -96.8475),   # Waxahachie, TX (I-35 solar corridor)
    "HB_SOUTH": (26.9070, -99.2715),   # Zapata, TX (South Texas inland wind belt - where projects actually are)
    "HB_WEST": (32.4518, -100.5371),   # Roscoe, TX ("Wind Energy Capital of Texas" - best West TX wind resource)
    "HB_HOUSTON": (29.3013, -94.7977), # Galveston, TX (Houston Hub's only wind project - excellent coastal wind resource)
    "HB_PAN": (35.2220, -101.8313),    # Amarillo, TX (Panhandle)
}

WIND_WEATHER_SOURCE_OPTIONS = {
    "Open-Meteo / PVGIS (Default)": "AUTO",
    "NOAA HRRR (Cached)": "NOAA_HRRR_CACHED",
}

WIND_MODEL_ENGINE_OPTIONS = {
    "Standard (Current)": "STANDARD",
    "Advanced Calibrated (EIA/CF/SCED/Node)": "ADVANCED_CALIBRATED",
    "V3 (Improved Curves + Per-Project Shear)": "V3_100M",
}

# Soft dispatch cap strength: 0.0 = ignore BP, 1.0 = hard cap at BP.
DEFAULT_BASE_POINT_CAP_STRENGTH = 0.30


def get_hrrr_cache_count():
    try:
        repo_root = Path(__file__).resolve().parent
        return len(list((repo_root / "data_cache" / "hrrr").glob("*.parquet")))
    except Exception:
        return 0


def estimate_base_point_headroom_factor(
    df,
    capacity_mw=None,
    quantile=0.90,
    min_factor=1.00,
    max_factor=1.35,
):
    """
    Estimate a transparent dispatch headroom multiplier from historical SCED:
    factor ~= quantile(Actual_MW / Base_Point_MW), clipped to sane bounds.
    """
    if df is None or len(df) == 0:
        return 1.0

    if "Actual_MW" not in df.columns or "Base_Point_MW" not in df.columns:
        return 1.0

    actual = pd.to_numeric(df["Actual_MW"], errors="coerce")
    bp = pd.to_numeric(df["Base_Point_MW"], errors="coerce")

    try:
        cap = float(capacity_mw)
    except Exception:
        cap = np.nan
    bp_floor = max(1.0, (cap * 0.02) if pd.notna(cap) and cap > 0 else 2.0)

    mask = bp > bp_floor
    if mask.sum() < 200:
        return 1.0

    ratio = (actual[mask] / bp[mask]).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return 1.0

    ratio = ratio.clip(lower=0.40, upper=2.00)
    factor = float(ratio.quantile(float(quantile)))
    factor = float(np.clip(factor, float(min_factor), float(max_factor)))
    return factor


@st.cache_data(ttl=3600, show_spinner=False)
def load_bill_data(file_path):
    """
    Loads generation data from the Azure Sky bill Excel file.
    Expects 'Date' column (datetime) and 'Plant Generation (MWh)' column.
    Returns a DataFrame with a timezone-aware DatetimeIndex (Central) and 'Actual_MW'.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return pd.DataFrame()

        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
            if "Time" not in df.columns:
                st.error(f"Bill parquet missing `Time`. Found: {df.columns.tolist()}")
                return pd.DataFrame()
            if "Actual_MW" not in df.columns:
                if "Plant Generation (MWh)" in df.columns:
                    df["Actual_MW"] = pd.to_numeric(df["Plant Generation (MWh)"], errors="coerce") * 4.0
                else:
                    st.error(f"Bill parquet missing `Actual_MW`. Found: {df.columns.tolist()}")
                    return pd.DataFrame()
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        else:
            df = pd.read_excel(path)

            # Validate columns
            req_cols = ['Date', 'Plant Generation (MWh)']
            if not all(col in df.columns for col in req_cols):
                st.error(f"Bill file missing columns. Found: {df.columns.tolist()}")
                return pd.DataFrame()

            # Parse Date/Time
            df['Time'] = pd.to_datetime(df['Date'])
            # Convert MWh to MW (15-min intervals => MW = MWh * 4)
            df['Actual_MW'] = pd.to_numeric(df['Plant Generation (MWh)'], errors='coerce') * 4.0
        
        # Localize/Convert to Central
        if df['Time'].dt.tz is None:
             df['Time'] = df['Time'].dt.tz_localize('US/Central', ambiguous='infer', nonexistent='shift_forward')
        else:
             df['Time'] = df['Time'].dt.tz_convert('US/Central')

        df['Actual_MW'] = pd.to_numeric(df['Actual_MW'], errors='coerce')
        
        # Set index
        df = df.set_index('Time').sort_index()
        
        # Extract just what we need
        return df[['Actual_MW']]
        
    except Exception as e:
        st.error(f"Error loading bill data: {e}")
        return pd.DataFrame()


def apply_base_point_cap(
    modeled_mw,
    base_point_mw,
    headroom_factor=1.0,
    capacity_mw=None,
    cap_strength=1.0,
):
    """
    Apply a soft Base Point cap to modeled MW.
    cap_strength=1.0 => hard cap at effective base point.
    cap_strength=0.0 => no cap.
    """
    modeled = pd.to_numeric(modeled_mw, errors="coerce").fillna(0.0)
    bp = pd.to_numeric(base_point_mw, errors="coerce")
    eff = bp * float(headroom_factor)
    if capacity_mw is not None:
        try:
            cap = float(capacity_mw)
            if cap > 0:
                eff = eff.clip(lower=0.0, upper=cap)
        except Exception:
            eff = eff.clip(lower=0.0)
    else:
        eff = eff.clip(lower=0.0)
    eff = eff.fillna(np.inf)
    capped = np.minimum(modeled, eff)
    w = float(np.clip(cap_strength, 0.0, 1.0))
    return (modeled * (1.0 - w)) + (capped * w)

# Page Config
st.set_page_config(page_title="VPPA Settlement Estimator", layout="wide")

# Custom CSS to widen sidebar (50% wider than default)
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 450px;
        max-width: 450px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("VPPA Settlement Estimator")
st.markdown("Compare multiple Virtual Power Purchase Agreement (VPPA) scenarios in ERCOT.")

if st.button("Hard Refresh App Cache", key="hard_refresh_app_cache"):
    st.cache_data.clear()
    st.session_state.pop("val_preview_results", None)
    st.session_state.pop("results", None)
    st.rerun()

# --- State Management ---
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []

# Wind defaults
st.session_state.setdefault("sb_wind_weather_source_label", "Open-Meteo / PVGIS (Default)")
st.session_state.setdefault("sb_wind_model_engine_label", "Standard (Current)")
st.session_state.setdefault("val_wind_weather_source_label", "Open-Meteo / PVGIS (Default)")
st.session_state.setdefault("val_wind_model_engine_label", "Standard (Current)")
st.session_state.setdefault("val_apply_bp_cap", True)
st.session_state.setdefault("bench_wind_weather_source_label", "Open-Meteo / PVGIS (Default)")
st.session_state.setdefault("bench_wind_model_engine_label", "Standard (Current)")



# Create Tabs
tab_guide, tab_validation, tab_scenarios, tab_performance, tab_vppa_8760 = st.tabs(
    [
        "📘 Guide & Business Context",
        "Bill Validation",
        "Scenario Analysis",
        "Model Performance",
        "VPPA 8760 Compare",
    ]
)

# --- Dynamic Sidebar Visibility ---
# Hide sidebar on Bill Validation, show on Scenario Analysis
components.html(
    """
    <script>
        // Use window.parent.document to access the main Streamlit document
        const doc = window.parent.document;
        
        function toggleSidebar() {
            const tabs = doc.querySelectorAll('button[data-testid="stTab"]');
            let hideSidebar = false;
            
            tabs.forEach(tab => {
                // Check for Guide, Validation, Performance, etc.
                if ((tab.innerText.includes("Guide") || tab.innerText.includes("Bill Validation") || tab.innerText.includes("Model Performance") || tab.innerText.includes("Weather Variability") || tab.innerText.includes("VPPA 8760 Compare")) && tab.getAttribute("aria-selected") === "true") {
                    hideSidebar = true;
                }
            });

            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                // If Bill Validation is active, hide. Otherwise show.
                sidebar.style.display = hideSidebar ? "none" : "";
            }
        }

        // Run frequently to handle tab switches and re-renders
        setInterval(toggleSidebar, 300);
    </script>
    """,
    height=0,
    width=0,
)

# --- Tab: Guide & Business Context ---
with tab_guide:
    st.header("📘 Guide & Business Context")
    
    # Load Guide Content from Markdown File
    guide_path = Path(__file__).resolve().parent / "docs" / "USER_GUIDE.md"
    if guide_path.exists():
        guide_text = guide_path.read_text(encoding="utf-8")
        st.markdown(guide_text)
        st.caption("Source of truth: docs/USER_GUIDE.md")
    else:
        st.warning("User guide file not found at docs/USER_GUIDE.md.")

    st.subheader("🗺️ User Workflow Chart")
    flow_labels = [
        "Start: Define Business Goal",
        "Price or Structure VPPA",
        "Validate Model vs Actual",
        "Review Model Quality",
        "Scenario Analysis",
        "Bill Validation",
        "Model Performance",
        "Set Assumptions",
        "Run Scenarios",
        "Review Metrics & Risk",
        "Export PDF/Excel",
        "Select Resource + Date Range",
        "Compare Model vs Actual",
        "Check Correlation/Errors",
        "Refine Assumptions",
        "Review Regional Benchmarks",
        "Pick Higher-Confidence Hubs",
    ]
    flow_source = [0, 0, 0, 1, 2, 3, 4, 7, 8, 9, 5, 11, 12, 13, 6, 15, 16]
    flow_target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 4]
    flow_value = [1] * len(flow_source)

    flow_fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=16,
                    thickness=18,
                    line=dict(color="rgba(70,70,70,0.5)", width=0.5),
                    label=flow_labels,
                    color=[
                        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#4E79A7",
                        "#E15759", "#76B7B2", "#59A14F", "#59A14F", "#59A14F",
                        "#59A14F", "#EDC948", "#EDC948", "#EDC948", "#EDC948",
                        "#B07AA1", "#B07AA1",
                    ],
                ),
                link=dict(
                    source=flow_source,
                    target=flow_target,
                    value=flow_value,
                    color="rgba(78,121,167,0.28)",
                ),
            )
        ]
    )
    flow_fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=12))
    st.plotly_chart(flow_fig, use_container_width=True, config={"displayModeBar": False})

    st.success("Ready to start? Open the Scenario Analysis tab to build your first case.")

with tab_scenarios:
    st.header("Scenario Analysis")
    st.markdown("Create and compare VPPA scenarios based on weather models. (Beta)")

    # Documentation Section
    with st.expander("📚 **Documentation: Data Sources & Methodology**", expanded=False):
        st.markdown("""
        ## Overview
        This tool estimates VPPA settlements by combining **actual ERCOT market prices** with **realistic renewable generation profiles** based on meteorological data.
        
        ---
        
        ## Data Sources
        
        ### 1. **Market Prices (ERCOT RTM)**
        - **Source:** [gridstatus.io](https://www.gridstatus.io/) - Real-time ERCOT data API
        - **Data:** 15-minute Real-Time Market (RTM) prices by settlement point
        - **Coverage:** 2020-2026 (historical actual prices)
        - **Hubs:** HB_NORTH, HB_SOUTH, HB_WEST, HB_HOUSTON, HB_PAN
        
        ### 2. **Generation Profiles (Wind/Solar)**
        - **Sources:** 
            - **[Open-Meteo](https://open-meteo.com/):** For **2024-2026 Actuals** (ERA5 Reanalysis). High-accuracy solar irradiance and 10m wind speeds.
            - **[PVGIS](https://re.jrc.ec.europa.eu/pvgis/):** For **History (2005-2023)** and **Typical Meteorological Year (TMY)** data.
        - **Method:**
          - **2024+:** Uses **Actual Open-Meteo Data** (Solar & Wind) ✅
          - **Historical Years (2005-2023):** Uses **Actual PVGIS Data** ✅
          - **TMY:** Only used when **"Force TMY"** checkbox is selected
        - **Sensitivity Analysis:**
          - Use the **"Force TMY"** checkbox to simulate "normal" weather conditions for any year, overriding actual weather data.
        
        ### 3. **Hub Location Coordinates**
        Based on analysis of **ERCOT project queue data** (787 renewable projects):
        
        | Hub | Location | Wind Resource | Rationale |
        |-----|----------|---------------|-----------|
        | **HB_NORTH** | Waxahachie, TX | 4.97 m/s @ 80m | I-35 solar corridor (192 projects) |
        | **HB_SOUTH** | Zapata, TX | 6.43 m/s @ 80m | South Texas inland wind belt (212 projects) |
        | **HB_WEST** | Roscoe, TX | 6.50 m/s @ 80m | "Wind Energy Capital of Texas" (252 projects) |
        | **HB_HOUSTON** | Galveston, TX | 7.47 m/s @ 80m | Coastal wind project location (6 projects) |
        | **HB_PAN** | Amarillo, TX | 6.44 m/s @ 80m | Texas Panhandle (44 projects) |
        
        ---
        
        ## Methodology
        
        ### VPPA Settlement Calculation
        ```
        For each 15-min interval:
        1. Generation Revenue = Generation (MWh) × Market Price ($/MWh)
        2. VPPA Payment = Generation (MWh) × VPPA Price ($/MWh)
        3. Net Settlement = Generation Revenue - VPPA Payment
        
        Monthly/Annual totals = Sum of all intervals
        ```
        
        ### Generation Profile Creation
        1. **Fetch Weather Data** from Open-Meteo (2024-2026) or PVGIS (History/TMY)
        2. **Convert to Power:**
           - Solar: GHI (Global Horizontal Irradiance) → DC power → inverter efficiency → AC MW
           - Wind: Wind speed (scaled to hub height) → power curve → MW
        3. **Resample** to 15-minute intervals
        4. **Align** timestamps to ERCOT Central Time
        
        ---
        
        ## Technical Details: Weather-to-Power Conversion
        
        ### Solar Generation Model
        
        **Input Data:**
        - **2024-2026 Actual:** Global Horizontal Irradiance (GHI) from Open-Meteo ERA5 reanalysis, W/m²
        - **Historical (2005-2023):** GHI from PVGIS (calculated as Gb(i) + Gd(i) + Gr(i) for horizontal plane)
        - **TMY:** GHI from PVGIS Typical Meteorological Year, W/m²
        
        **Conversion Formula:**
        ```
        Solar_MW = Capacity_MW × (GHI / 1000) × System_Efficiency
        
        Where:
        - GHI is in W/m²
        - System_Efficiency = 0.85 (accounts for DC-to-AC conversion, soiling, temperature losses)
        - Output is clipped at Capacity_MW (no overgeneration)
        ```
        
        **Key Assumptions:**
        - **Panel Orientation:** Horizontal tracking (simplification - actual projects use tilted/tracking)
        - **Performance Ratio:** 85% accounts for:
          - Inverter losses (~3%)
          - Temperature derating (~5%)
          - Soiling/shading (~4%)
          - Wiring/mismatch (~3%)
        - **Capacity Factor:** Typical range 20-25% in Texas
        
        ---
        
        ### Wind Generation Model
        
        **Input Data:**
        - **All Sources:** 10-meter wind speed (m/s) from Open-Meteo or PVGIS
        - **Note:** We use 10m data consistently across all years for methodology alignment
        
        **Step 1: Extrapolate to Hub Height (80m)**
        
        Wind speed increases with height following a power law. We apply empirically-tuned scaling factors:
        
        ```
        Wind_Speed_80m = Wind_Speed_10m × Shear_Factor
        
        Shear_Factor by Region:
        - East Texas / Houston (lon > -96.0°): 1.60
        - West / South / Panhandle (lon ≤ -96.0°): 1.95
        ```
        
        **Why Regional Scaling?**
        - **Coastal (Houston):** Lower surface roughness → lower shear exponent
        - **Inland (West/South/Pan):** Higher terrain roughness → higher shear exponent
        - These factors were calibrated against EIA-923 actual generation data
        
        **Step 2: Apply Power Curve**
        
        We use a simplified IEC Class 2 turbine power curve:
        
        ```
        Normalized_Power = 
            0.0                           if v < 3.0 m/s   (cut-in speed)
            ((v - 3.0) / 9.0)³           if 3.0 ≤ v < 12.0 m/s   (cubic region)
            1.0                           if 12.0 ≤ v < 25.0 m/s  (rated power)
            0.0                           if v ≥ 25.0 m/s  (cut-out speed)
        
        Wind_MW = Normalized_Power × Capacity_MW
        ```
        
        **Key Assumptions:**
        - **Turbine Type:** Generic 2.5-3.5 MW turbine (representative of Texas fleet)
        - **Hub Height:** 80 meters (typical for modern Texas wind farms)
        - **Cut-in Speed:** 3 m/s (turbine starts generating)
        - **Rated Speed:** 12 m/s (full power output)
        - **Cut-out Speed:** 25 m/s (turbine shuts down for safety)
        - **Capacity Factor:** Typical range 35-45% in good Texas wind sites
        
        **Power Curve Shape:**
        - **Cubic relationship** in the 3-12 m/s range reflects physics: Power ∝ v³
        - This is the most sensitive region where small wind speed errors have large generation impacts
        
        ---
        
        ### Data Processing Pipeline
        
        1. **Fetch:** Hourly weather data (8,760 or 8,784 points for leap years)
        2. **Convert:** Apply solar or wind model → hourly MW profile
        3. **Interpolate:** Resample from hourly to 15-minute intervals using linear interpolation
        4. **Align:** Match timestamps to ERCOT market data (UTC → Central Time)
        5. **Validate:** Ensure 35,040 or 35,136 intervals (15-min resolution for full year)
        
        ### Curtailment Modeling
        - **Default:** Negative prices ($<0) are floored at $0 (curtailment)
        - **Optional:** "No Curtailment" mode keeps negative prices (financial exposure)
        
        ---
        
        ## Validation
        
        **Validated against EIA-923 actual generation data (2024):**
        - ✅ Seasonal patterns match (Spring peak, Summer low)
        - ✅ 222 Texas wind plants: 124.3 TWh actual vs our synthetic profiles
        - ✅ Month-to-month relative changes accurate
        
        ---
        
        ## Limitations
        
        - **Transmission costs not included**
        - **Basis risk** (hub vs project location) simplified
        - **Synthetic profiles** represent typical conditions, actual may vary ±20%
        - **Future market prices** use historical data (not forecasts)
        
        ---
        
        ## Custom Profile Upload
        - **Format:** CSV with `Gen_MW` column (hourly 8760 or 15-min 35,040 rows)
        - **Timezone:** Assumes UTC if not specified, converts to Central
        - **Leap years:** Automatically handled (8,784 hourly or 35,136 15-min rows)
        """)

    # Main Scenario View (Using the logic that was here)
    if not st.session_state.scenarios:
        st.info("👈 Use the sidebar to create your first scenario!")
    
    # ... (Rest of the main content will be processed in subsequent steps)



    

@st.cache_data(show_spinner=False, ttl=3600)
def get_ercot_data(year, cache_token=None):
    cache_file = f"ercot_rtm_{year}.parquet"

    # Try loading from local file first
    try:
        if pd.io.common.file_exists(cache_file):
            df = pd.read_parquet(cache_file)
            return df
    except Exception as e:
        st.warning(f"Could not load local cache: {e}")

    iso = gridstatus.Ercot()
    try:
        # st.spinner is not needed inside cache, but we can use st.status in main app
        df = iso.get_rtm_spp(year=year)
        
        # Pre-process: Ensure Time is datetime and localized
        if not pd.api.types.is_datetime64_any_dtype(df['Time']):
            df['Time'] = pd.to_datetime(df['Time'], utc=True)
        
        # Create Central Time column
        df['Time_Central'] = df['Time'].dt.tz_convert('US/Central')
        
        # Memory Optimization: Downcast float64 to float32
        # Prices and other metrics don't need 64-bit precision
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Save to local parquet for future speedups
        try:
            df.to_parquet(cache_file)
        except Exception as e:
            st.warning(f"Could not save local cache: {e}")
        
        return df
    except Exception as e:
        # Raise instead of returning empty so a transient fetch error is not cached as valid data.
        raise RuntimeError(f"Error fetching data for {year}: {e}") from e


def load_market_data(year):
    """Loads ERCOT market data with cache invalidation and safe UI error reporting."""
    cache_path = f"ercot_rtm_{year}.parquet"
    if os.path.exists(cache_path):
        stat = os.stat(cache_path)
        # Include nanosecond mtime + size so Streamlit cache invalidates on file updates.
        cache_token = (int(stat.st_mtime_ns), int(stat.st_size))
    else:
        cache_token = (0, 0)
    try:
        return get_ercot_data(year, cache_token=cache_token)
    except Exception as e:
        st.error(str(e))
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def load_market_hub_data(year, hub):
    """
    Return a minimal, hub-filtered market frame for fast bill validation and preview joins.
    """
    df_market = load_market_data(year)
    if df_market.empty:
        return pd.DataFrame(columns=["Time", "Time_Central", "SPP"])

    df_market_hub = df_market.loc[
        df_market["Location"] == hub,
        ["Time", "Time_Central", "SPP"],
    ].copy()

    if df_market_hub.empty:
        return df_market_hub

    # Normalize timezone once so downstream logic doesn't repeat conversion work.
    if df_market_hub["Time"].dt.tz is None:
        df_market_hub["Time"] = df_market_hub["Time"].dt.tz_localize("UTC")
    else:
        df_market_hub["Time"] = df_market_hub["Time"].dt.tz_convert("UTC")

    if df_market_hub["Time_Central"].dt.tz is None:
        df_market_hub["Time_Central"] = df_market_hub["Time_Central"].dt.tz_localize("US/Central")
    else:
        df_market_hub["Time_Central"] = df_market_hub["Time_Central"].dt.tz_convert("US/Central")

    return df_market_hub.sort_values("Time").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=900)
def parse_uploaded_bill_file(file_bytes, file_extension):
    """
    Parse an uploaded bill file into a DataFrame.
    Cached by file contents to avoid reprocessing on each Streamlit rerun.
    """
    file_buffer = io.BytesIO(file_bytes)

    if file_extension == "csv":
        return pd.read_csv(file_buffer)

    if file_extension in {"xlsx", "xls"}:
        engine = "openpyxl" if file_extension == "xlsx" else None
        return pd.read_excel(file_buffer, engine=engine)

    if file_extension == "pdf":
        import pdfplumber

        best_table = None
        with pdfplumber.open(file_buffer) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table and len(table) > 1:
                    if best_table is None or len(table) > len(best_table):
                        best_table = table
                    # Stop early when we already have a sizable table.
                    if len(best_table) >= 100:
                        break
                    continue

                if best_table is None:
                    for candidate in page.extract_tables() or []:
                        if candidate and len(candidate) > 1:
                            if best_table is None or len(candidate) > len(best_table):
                                best_table = candidate

        if not best_table:
            raise ValueError("No tables found in PDF. Ensure the PDF contains tabular data.")

        headers = [
            (str(h).strip() if h is not None else "")
            for h in best_table[0]
        ]
        if not any(headers):
            headers = [f"col_{i}" for i in range(len(headers))]

        rows = best_table[1:]
        return pd.DataFrame(rows, columns=headers)

    raise ValueError(f"Unsupported file type: {file_extension}")


@st.cache_data(show_spinner=False)
def get_cached_asset_data(resource_id, start_date, end_date):
    """Cached wrapper for SCED fetching to prevent re-loading on every interaction."""
    return sced_fetcher.get_asset_period_data(resource_id, start_date, end_date)


@st.cache_data(show_spinner=False)
def get_cached_asset_data_with_base_point(resource_id, start_date, end_date):
    """
    Cached SCED loader that prefers Base Point-enriched data but degrades safely
    to regular cached SCED when Base Point is unavailable.
    """
    df_bp = sced_fetcher.get_asset_period_data(
        resource_id,
        start_date,
        end_date,
        require_base_point=True,
    )
    if not df_bp.empty:
        return df_bp
    return sced_fetcher.get_asset_period_data(resource_id, start_date, end_date)


@st.cache_data(show_spinner=False)
def get_cached_asset_data_with_base_point_local(resource_id, start_date, end_date):
    """
    Cache-only SCED loader for interactive validation flows.
    Avoids live ERCOT pulls that can stall the UI for long periods.
    """
    df_bp = sced_fetcher.get_asset_period_data_cache_only(
        resource_id,
        start_date,
        end_date,
        require_base_point=True,
    )
    if not df_bp.empty:
        return df_bp
    return sced_fetcher.get_asset_period_data_cache_only(resource_id, start_date, end_date)


@st.cache_data(show_spinner=False, ttl=900)
def get_first_cached_sced_date(resource_id):
    """
    Return the first local SCED date available for a resource, if any.
    Uses daily cache filenames and falls back to consolidated year files.
    """
    if not resource_id:
        return None

    rid = str(resource_id).strip()
    if not rid:
        return None

    cache_dir = REPO_ROOT / sced_fetcher.CACHE_DIR
    if not cache_dir.exists():
        return None

    first_date = None
    for p in cache_dir.glob(f"????-??-??_{rid}.parquet"):
        try:
            d = datetime.strptime(p.name[:10], "%Y-%m-%d").date()
        except Exception:
            continue
        if first_date is None or d < first_date:
            first_date = d

    if first_date is not None:
        return first_date.isoformat()

    # Fallback: some resources may only have consolidated year cache files.
    for p in sorted(cache_dir.glob(f"{rid}_*_full.parquet")):
        try:
            df_time = pd.read_parquet(p, columns=["Time"])
            t = pd.to_datetime(df_time["Time"], utc=True, errors="coerce").dropna()
            if not t.empty:
                d = t.min().date()
                if first_date is None or d < first_date:
                    first_date = d
        except Exception:
            continue

    return first_date.isoformat() if first_date is not None else None

# --- Helper Functions ---
def calculate_scenario(scenario, df_rtm):
    """Calculates settlement for a single scenario."""
    # Filter by Hub
    df_hub = df_rtm[df_rtm['Location'] == scenario['hub']].copy()
    
    # Filter by Date (if needed, currently full year)
    if scenario.get('duration') == 'Specific Month':
        month_map = {
            "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
            "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
        }
        target_month = month_map.get(scenario.get('month'))
        if target_month:
            # Filter to specific month
            df_hub = df_hub[df_hub['Time_Central'].dt.month == target_month].copy()
            
            # If this is the current month/year, truncate to available data
            current_date = pd.Timestamp.now(tz='US/Central')
            scenario_year = scenario['year']
            
            if scenario_year == current_date.year and target_month == current_date.month:
                # This is the current month - limit to available data
                # Use the max date actually in the data or today, whichever is earlier
                if not df_hub.empty:
                    max_available = df_hub['Time_Central'].max()
                    cutoff = min(max_available, current_date)
                    df_hub = df_hub[df_hub['Time_Central'] <= cutoff].copy()
                    
                    # Store the actual date range used for display
                    scenario['actual_end_date'] = cutoff
                    scenario['date_range_note'] = f"Month-to-date through {cutoff.strftime('%b %d, %Y')}"
    
    # Robustness: Handle empty dataframe
    if df_hub.empty:
        # Return empty dataframe with expected columns
        empty_df = pd.DataFrame(columns=['Time_Central', 'Potential_Gen_MW', 'VPPA_Price', 'Settlement_Price', 'Actual_Gen_MW', 
                                       'Gen_Energy_MWh', 'Curtailed_MWh', 'Settlement_Amount', 'Cumulative_Settlement', 'SPP'])
        # Ensure correct dtypes
        for col in empty_df.columns:
            if col == 'Time_Central':
                empty_df[col] = pd.to_datetime(empty_df[col])
            else:
                empty_df[col] = pd.to_numeric(empty_df[col])
        return empty_df

    # Generate Profile using TMY Data
    # Note: We use the scenario year to align the TMY data to the correct timestamps
    interval_hours = 0.25
    capacity_mw = scenario['capacity_mw']
    tech = scenario['tech']
    
    # Default to Abilene if hub not found
    default_loc = (32.4487, -99.7331)  # Abilene, TX
    
    # Check for custom location override
    if scenario.get('custom_lat') is not None and scenario.get('custom_lon') is not None:
        lat, lon = scenario['custom_lat'], scenario['custom_lon']
    else:
        lat, lon = HUB_LOCATIONS.get(scenario['hub'], default_loc)

    try:
        # --- Handle Special Sources (BILL / SCED) ---
        source_type = scenario.get('source_type', 'MODEL')
        
        if source_type == 'BILL':
            bill_path = scenario.get('bill_file_path')
            if bill_path:
                df_bill = load_bill_data(bill_path)
                if not df_bill.empty:
                    # Align to Central Time and Hub Timestamps
                    # Assuming bill data is already Central (handled in loader)
                    profile_central = df_bill['Actual_MW']
                else:
                    raise ValueError(f"Bill data empty or not found: {bill_path}")
            else:
                 raise ValueError("Scenario marked as BILL source but no file path provided.")
                 
        elif source_type == 'SCED':
            res_id = scenario.get('resource_id')
            if not res_id:
                # Fallback mapping if not explicit (e.g. from older scenario)
                if scenario['tech'] == 'Wind' and 'Azure' in scenario['name']:
                    res_id = "AZURE_SKY_WIND_AGG"
                else:
                    # Try to find a default or error?
                    # For now, if no resource ID, fall back to MODEL
                    pass
            
            if res_id:
                # Fetch SCED Data
                # Function signature: get_asset_period_data(resource_name, start_date, end_date)
                # We need whole year for the scenario
                year = scenario['year']
                start_date = datetime(year, 1, 1).date()
                end_date = datetime(year, 12, 31).date()
                
                # Fetch (this uses cache)
                df_sced = sced_fetcher.get_asset_period_data(res_id, start_date, end_date)
                
                if not df_sced.empty:
                     # Ensure Time index
                     if 'Time' in df_sced.columns:
                         df_sced['Time'] = pd.to_datetime(df_sced['Time'])
                         if df_sced['Time'].dt.tz is None:
                             df_sced = df_sced.set_index(df_sced['Time'].dt.tz_localize('UTC').dt.tz_convert('US/Central'))
                         else:
                             df_sced = df_sced.set_index(df_sced['Time'].dt.tz_convert('US/Central'))
                     
                     # Extract Actual_MW
                     # Handle scaling if capacity differs?
                     # SCED is actual MW. If user changed capacity in UI, do we scale?
                     # Ideally yes: (User Cap / Asset Cap) * Actual MW
                     # Only if asset capacity is known.
                     asset_cap = scenario.get('asset_capacity_mw', 350.0)
                     user_cap = scenario.get('capacity_mw', 350.0)
                     
                     scale_factor = 1.0
                     if asset_cap and asset_cap > 0:
                         scale_factor = user_cap / asset_cap
                         
                     profile_central = df_sced['Actual_MW'] * scale_factor
                else:
                    st.warning(f"No SCED data found for {res_id} in {year}. Falling back to Model.")
                    source_type = 'MODEL' # Fallback

        if source_type == 'MODEL' or 'profile_central' not in locals():
            if tech == "Custom Upload":
                # Load Custom CSV
                csv_path = scenario.get('custom_profile_path')
                if csv_path and pd.io.common.file_exists(csv_path):
                    df_custom = pd.read_csv(csv_path)
                    
                    # Normalize columns
                    df_custom.columns = [c.lower().strip() for c in df_custom.columns]
                    
                    # Identify MW column
                    mw_col = next((c for c in df_custom.columns if any(x in c for x in ['mw', 'gen', 'load', 'power'])), None)
                    if not mw_col:
                        raise ValueError("Could not identify Generation/MW column in CSV.")
                    
                    # Identify Time column
                    time_col = next((c for c in df_custom.columns if any(x in c for x in ['time', 'date', 'hour'])), None)
                    
                    if time_col:
                        # Parse Time
                        df_custom['Time'] = pd.to_datetime(df_custom[time_col], utc=True) # Assume UTC if not specified? 
                        # If naive, assume UTC or Central? Let's assume input matches expected years or generic.
                        # Best effort: convert to Central if possible, or just naive.
                        if df_custom['Time'].dt.tz is None:
                            df_custom['Time'] = df_custom['Time'].dt.tz_localize('UTC') # Assume UTC
                        
                        profile_series = df_custom.set_index('Time')[mw_col]
                        profile_central = profile_series.tz_convert('US/Central')
                    else:
                        # Infer Time Index based on length
                        year = scenario['year']
                        start_date = f"{year}-01-01"
                        
                        if len(df_custom) >= 35000: # Approx 15-min (35040)
                            freq = '15min'
                        else: # Assume Hourly (8760/8784)
                            freq = 'h'
                        
                        # Create index
                        idx = pd.date_range(start=start_date, periods=len(df_custom), freq=freq, tz='US/Central')
                        profile_central = pd.Series(df_custom[mw_col].values, index=idx)

            else:
                raise FileNotFoundError(
                    f"Custom profile not found for scenario '{scenario.get('name', 'Unnamed')}'. "
                    "Re-upload the profile CSV and re-run."
                )

        else:
            # Standard TMY/Actual Logic
            # Extract override flag
            force_tmy = scenario.get('force_tmy', False)
            
            # Ensure module is reloaded to pick up recent changes (hotfix)
            import importlib
            importlib.reload(fetch_tmy)
            
            if force_tmy:
                st.toast(f"ℹ️ Forcing TMY Data for {scenario['name']}")
            
            profile_series = fetch_tmy.get_profile_for_year(
                year=scenario['year'],
                tech=tech,
                capacity_mw=capacity_mw,
                lat=lat,
                lon=lon,
                force_tmy=force_tmy,
                hub_name=scenario.get('hub'),
                apply_wind_calibration=(tech == "Wind"),
                wind_weather_source=scenario.get("wind_weather_source", "AUTO"),
                hrrr_forecast_hour=int(scenario.get("hrrr_forecast_hour", 0)),
                wind_model_engine=scenario.get("wind_model_engine", "STANDARD"),
            )
            # Align profile with df_hub timestamps
            profile_central = profile_series.tz_convert('US/Central')
        
        if profile_central is not None:
             # Reindex to match df_hub['Time_Central']
             potential_gen = profile_central.reindex(df_hub['Time_Central'], method='nearest').values
        else:
             potential_gen = np.zeros(len(df_hub))
        
    except Exception as e:
        raise RuntimeError(
            f"Profile generation failed for scenario '{scenario.get('name', 'Unnamed')}': {e}"
        ) from e

    df_hub['Potential_Gen_MW'] = potential_gen

    if tech == "Wind" and not df_hub.empty:
        potential_with_haircut = apply_congestion_haircut(
            gen_series=pd.Series(df_hub["Potential_Gen_MW"].values, index=df_hub.index),
            spp_series=df_hub["SPP"],
            hub_name=scenario.get("hub"),
            resource_id=scenario.get("resource_id"),
        )
        df_hub["Potential_Gen_MW"] = potential_with_haircut.values
    
    # Settlement
    vppa_price = scenario.get('vppa_price', scenario.get('strike_price', 50.0))
    df_hub['VPPA_Price'] = vppa_price
    
    # Revenue Share: If > 0, buyer only gets that percentage of the upside when SPP > VPPA price
    revenue_share_pct = scenario.get('revenue_share_pct', 100) / 100.0  # Convert from percentage to decimal
    
    if revenue_share_pct < 1.0:
        # When SPP > VPPA: Settlement = (SPP - VPPA) * share_pct (buyer gets only their share of upside)
        # When SPP <= VPPA: Settlement = SPP - VPPA (full downside, no sharing)
        upside = np.maximum(df_hub['SPP'] - vppa_price, 0)  # Positive when SPP > VPPA
        downside = np.minimum(df_hub['SPP'] - vppa_price, 0)  # Negative when SPP < VPPA
        df_hub['Settlement_Price'] = (upside * revenue_share_pct) + downside
    else:
        df_hub['Settlement_Price'] = df_hub['SPP'] - vppa_price
        
    # Negative Price Floor Logic
    # If the generator stops losing money below a certain negative SPP (e.g. they curtail or have a hedge floor)
    floor_price = scenario.get('negative_price_floor')
    if floor_price is not None:
        # We only apply the floor to the SPP *before* calculating the settlement against the strike
        # Example: Strike is $40. SPP is -$20. Floor is -$3. 
        # Without floor: Settlement = -$20 - $40 = -$60
        # With floor: Effective SPP = -$3. Settlement = -$3 - $40 = -$43.
        # This means the project's worst case SPP is capped at the floor.
        effective_spp = np.maximum(df_hub['SPP'], floor_price)
        if revenue_share_pct < 1.0:
            upside = np.maximum(effective_spp - vppa_price, 0)
            downside = np.minimum(effective_spp - vppa_price, 0)
            df_hub['Settlement_Price'] = (upside * revenue_share_pct) + downside
        else:
            df_hub['Settlement_Price'] = effective_spp - vppa_price
            
    
    # Curtailment
    if scenario.get('no_curtailment'):
        df_hub['Actual_Gen_MW'] = df_hub['Potential_Gen_MW']
    else:
        df_hub['Actual_Gen_MW'] = np.where(df_hub['SPP'] < 0, 0.0, df_hub['Potential_Gen_MW'])
    
    # Financials
    df_hub['Gen_Energy_MWh'] = df_hub['Actual_Gen_MW'] * interval_hours
    df_hub['Curtailed_MWh'] = (df_hub['Potential_Gen_MW'] - df_hub['Actual_Gen_MW']) * interval_hours
    df_hub['Settlement_Amount'] = df_hub['Settlement_Price'] * df_hub['Gen_Energy_MWh']
    df_hub['Cumulative_Settlement'] = df_hub['Settlement_Amount'].cumsum()
    
    return df_hub

def _build_settlement_excel(results: list, df_summary: pd.DataFrame, scenarios: list) -> bytes:
    """Build a well-formatted Excel workbook for the VPPA Settlement Estimator."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book

        # ── Shared formats ────────────────────────────────────────────────
        hdr_fmt = wb.add_format({
            "bold": True, "bg_color": "#0171BB", "font_color": "#FFFFFF",
            "border": 1, "text_wrap": True, "valign": "vcenter",
        })
        dollar_fmt   = wb.add_format({"num_format": "$#,##0", "border": 1})
        dollar2_fmt  = wb.add_format({"num_format": "$#,##0.00", "border": 1})
        number_fmt   = wb.add_format({"num_format": "#,##0", "border": 1})
        number1_fmt  = wb.add_format({"num_format": "#,##0.0", "border": 1})
        pct_fmt      = wb.add_format({"num_format": "0.0%", "border": 1})
        text_fmt     = wb.add_format({"border": 1})
        bold_fmt     = wb.add_format({"bold": True, "border": 1})
        title_fmt    = wb.add_format({"bold": True, "font_size": 14})
        subtitle_fmt = wb.add_format({"bold": True, "font_size": 11, "bottom": 1})

        # ══════════════════════════════════════════════════════════════════
        # Sheet 1 — Summary
        # ══════════════════════════════════════════════════════════════════
        ws = wb.add_worksheet("Summary")
        writer.sheets["Summary"] = ws
        ws.hide_gridlines(2)
        ws.set_column("A:A", 22)
        ws.set_column("B:G", 18)

        ws.write("A1", "VPPA Settlement Summary", title_fmt)
        ws.write("A2", f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}", wb.add_format({"italic": True, "font_color": "#666666"}))

        row = 4
        cols = ["Scenario", "Net Settlement ($)", "Total Gen (MWh)",
                "Curtailed (MWh)", "Capture Price ($/MWh)", "Avg Hub Price ($/MWh)"]
        col_fmts = [text_fmt, dollar_fmt, number_fmt, number_fmt, dollar2_fmt, dollar2_fmt]

        for c, col_name in enumerate(cols):
            ws.write(row, c, col_name, hdr_fmt)
        for r, res_row in df_summary.iterrows():
            row += 1
            for c, col_name in enumerate(cols):
                ws.write(row, c, res_row[col_name], col_fmts[c])

        # ── Scenario config table below summary ──────────────────────────
        row += 3
        ws.write(row, 0, "Scenario Configuration", subtitle_fmt)
        ws.merge_range(row, 0, row, 5, "Scenario Configuration", subtitle_fmt)
        row += 1
        cfg_cols = ["Scenario", "Year", "Hub", "Technology", "Capacity (MW)", "VPPA Price ($/MWh)"]
        for c, col_name in enumerate(cfg_cols):
            ws.write(row, c, col_name, hdr_fmt)
        for scen in scenarios:
            row += 1
            ws.write(row, 0, scen.get("name", ""), text_fmt)
            ws.write(row, 1, scen.get("year", ""), text_fmt)
            ws.write(row, 2, scen.get("hub", ""), text_fmt)
            ws.write(row, 3, scen.get("tech", ""), text_fmt)
            ws.write(row, 4, scen.get("capacity_mw", 0), number1_fmt)
            ws.write(row, 5, scen.get("vppa_price", 0), dollar2_fmt)

        # ══════════════════════════════════════════════════════════════════
        # Sheet 2 — Monthly Breakdown (one sub-table per scenario)
        # ══════════════════════════════════════════════════════════════════
        ws2 = wb.add_worksheet("Monthly")
        writer.sheets["Monthly"] = ws2
        ws2.hide_gridlines(2)
        ws2.set_column("A:A", 12)
        ws2.set_column("B:E", 18)

        ws2.write("A1", "Monthly Breakdown", title_fmt)
        row2 = 2

        for res in results:
            m_agg = res.get("monthly_agg")
            if m_agg is None or m_agg.empty:
                continue
            m_agg = m_agg.sort_values("Month_Num")

            ws2.write(row2, 0, res["Scenario"], subtitle_fmt)
            ws2.merge_range(row2, 0, row2, 4, res["Scenario"], subtitle_fmt)
            row2 += 1

            m_cols = ["Month", "Settlement ($)", "Generation (MWh)", "Avg $/MWh"]
            for c, h in enumerate(m_cols):
                ws2.write(row2, c, h, hdr_fmt)

            total_settle = 0.0
            total_gen = 0.0
            for _, mrow in m_agg.iterrows():
                row2 += 1
                settle = float(mrow.get("Settlement_Amount", 0))
                gen = float(mrow.get("Gen_Energy_MWh", 0))
                avg_rate = settle / gen if gen else 0
                total_settle += settle
                total_gen += gen
                ws2.write(row2, 0, mrow.get("Month", ""), text_fmt)
                ws2.write(row2, 1, settle, dollar_fmt)
                ws2.write(row2, 2, gen, number_fmt)
                ws2.write(row2, 3, avg_rate, dollar2_fmt)

            # Totals row
            row2 += 1
            ws2.write(row2, 0, "Total", bold_fmt)
            ws2.write(row2, 1, total_settle, wb.add_format({"num_format": "$#,##0", "border": 1, "bold": True, "top": 6}))
            ws2.write(row2, 2, total_gen, wb.add_format({"num_format": "#,##0", "border": 1, "bold": True, "top": 6}))
            avg_total = total_settle / total_gen if total_gen else 0
            ws2.write(row2, 3, avg_total, wb.add_format({"num_format": "$#,##0.00", "border": 1, "bold": True, "top": 6}))
            row2 += 3

        # ══════════════════════════════════════════════════════════════════
        # Sheet 3 — Daily Cumulative
        # ══════════════════════════════════════════════════════════════════
        ws3 = wb.add_worksheet("Daily Cumulative")
        writer.sheets["Daily Cumulative"] = ws3
        ws3.hide_gridlines(2)

        # Build a wide-format table: Date | Scenario1 | Scenario2 | ...
        daily_frames = {}
        for res in results:
            d_agg = res.get("daily_agg")
            if d_agg is None or d_agg.empty:
                continue
            s = d_agg.set_index("Time_Central")["Settlement_Amount"]
            s.name = res["Scenario"]
            daily_frames[res["Scenario"]] = s

        if daily_frames:
            df_daily_wide = pd.DataFrame(daily_frames)
            df_daily_wide.index.name = "Date"
            df_daily_wide = df_daily_wide.sort_index()

            ws3.write("A1", "Daily Cumulative Settlement ($)", title_fmt)
            row3 = 2

            ws3.set_column(0, 0, 14)
            ws3.set_column(1, len(daily_frames), 18)

            # Header
            date_hdr_fmt = wb.add_format({"bold": True, "bg_color": "#0171BB", "font_color": "#FFFFFF", "border": 1, "num_format": "yyyy-mm-dd"})
            ws3.write(row3, 0, "Date", hdr_fmt)
            for c, scen_name in enumerate(df_daily_wide.columns, start=1):
                ws3.write(row3, c, scen_name, hdr_fmt)

            date_fmt = wb.add_format({"num_format": "yyyy-mm-dd", "border": 1})
            for _, (dt, vals) in enumerate(df_daily_wide.iterrows()):
                row3 += 1
                ws3.write_datetime(row3, 0, dt.to_pydatetime(), date_fmt)
                for c, val in enumerate(vals, start=1):
                    ws3.write(row3, c, val if pd.notna(val) else "", dollar_fmt)

    return buf.getvalue()


def generate_pdf_report(results, df_summary):
    """Generate a simpler PDF report with summary metrics, without requiring Kaleido for charts."""
    
    pdf_buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#0171BB'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#0171BB'),
        spaceAfter=12,
        spaceBefore=12
    )
    normal_style = styles['Normal']
    
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph("VPPA Settlement Analysis Report", title_style))
    elements.append(Spacer(1, 0.5*inch))
    
    report_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    elements.append(Paragraph(f"<b>Generated:</b> {report_date}", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    num_scenarios = len(results)
    elements.append(Paragraph(f"<b>Number of Scenarios:</b> {num_scenarios}", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    if results:
        best_scenario = max(results, key=lambda x: x['Net Settlement ($)'])
        elements.append(Paragraph(
            f"<b>Best Performer:</b> {best_scenario['Scenario']}<br/>"
            f"Net Settlement: ${best_scenario['Net Settlement ($)']:,.0f}",
            normal_style
        ))
    
    elements.append(PageBreak())
    
    # --- Summary Metrics Table ---
    elements.append(Paragraph("Summary Metrics", heading_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Prepare table data
    table_data = [['Scenario', 'Net Settlement', 'Total Gen', 'Curtailed', 'Capture Price', 'Avg Hub Price']]
    
    for _, row in df_summary.iterrows():
        table_data.append([
            Paragraph(str(row['Scenario']), ParagraphStyle('Small', fontSize=8)),
            f"${row['Net Settlement ($)']:,.0f}",
            f"{row['Total Gen (MWh)']:,.0f}",
            f"{row['Curtailed (MWh)']:,.0f}",
            f"${row['Capture Price ($/MWh)']:.2f}",
            f"${row['Avg Hub Price ($/MWh)']:.2f}"
        ])
    
    # Create table
    table = Table(table_data, colWidths=[2.2*inch, 1.1*inch, 0.9*inch, 0.9*inch, 1*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0171BB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(table)
    elements.append(PageBreak())
    
    # --- Scenario Configuration Details ---
    elements.append(Paragraph("Scenario Configuration", heading_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Prepare Table Data
    # Columns: Scenario Name, Year, Hub, Tech, Capacity (MW), VPPA Price ($)
    scen_data = [['Scenario', 'Yr', 'Hub', 'Tech', 'Cap(MW)', 'Price($)']]
    
    for res in results:
        # Handle potential missing keys if using old session state, though rerun should fix
        scen_data.append([
            Paragraph(str(res.get('Scenario', '')), ParagraphStyle('Small', fontSize=8)),
            str(res.get('Year', '-')),
            str(res.get('Hub', '-')).replace('HB_', ''), # Shorten hub name
            str(res.get('Tech', '-')),
            f"{res.get('Capacity (MW)', 0):.1f}",
            f"{res.get('VPPA Price ($/MWh)', 0):.2f}"
        ])
        
    scen_table = Table(scen_data, colWidths=[2.5*inch, 0.6*inch, 1.0*inch, 0.8*inch, 0.8*inch, 0.8*inch])
    scen_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0171BB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
    ]))
    
    elements.append(scen_table)
    elements.append(PageBreak())

    # --- Monthly Performance Data ---
    elements.append(Paragraph("Monthly Performance Details", heading_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Prepare monthly data table
    # Columns: Scenario, Month, Net Settlement, Generation, Price
    monthly_rows = [['Scenario', 'Month', 'Settlement ($)', 'Gen (MWh)']]
    
    # Aggregate all monthly rows
    for res in results:
        if 'monthly_agg' in res and not res['monthly_agg'].empty:
            m_df = res['monthly_agg'].sort_values('Month_Num')
            # Sort by Month Num if possible, it's already sorted by default usually
            for _, row in m_df.iterrows():
                monthly_rows.append([
                    Paragraph(str(res['Scenario']), ParagraphStyle('Tiny', fontSize=7)),
                    str(row['Month']),
                    f"${row['Settlement_Amount']:,.0f}",
                    f"{row['Gen_Energy_MWh']:,.0f}"
                ])
    
    # If too many rows, this might span multiple pages which ReportLab handles automatically with Table,
    # but big tables can be tricky. `Table` flowable splits automatically in SimpleDocTemplate? 
    # Yes, it should split if using `LongTable` or if `Table` is smart enough (SplitTable). 
    # ReportLab's standard Table does split across pages.
    
    # Use smaller font for this dense table
    m_table = Table(monthly_rows, colWidths=[3*inch, 1*inch, 1.5*inch, 1*inch], repeatRows=1)
    m_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0171BB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'), # Align scenario name left
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    
    elements.append(m_table)
    elements.append(Spacer(1, 0.5*inch))
    
    # --- Key Insights ---
    elements.append(Paragraph("Key Insights", heading_style))
    elements.append(Spacer(1, 0.2*inch))
    
    if len(results) > 1:
        final_settlements = {r['Scenario']: r['Net Settlement ($)'] for r in results}
        best_scen = max(final_settlements, key=final_settlements.get)
        best_val = final_settlements[best_scen]
        worst_scen = min(final_settlements, key=final_settlements.get)
        worst_val = final_settlements[worst_scen]
        
        insights = [
            f"• <b>Best Performing:</b> {best_scen} with ${best_val:,.0f}",
            f"• <b>Lowest Performing:</b> {worst_scen} with ${worst_val:,.0f}",
            f"• <b>Performance Spread:</b> ${best_val - worst_val:,.0f}",
        ]
        
        for insight in insights:
            elements.append(Paragraph(insight, normal_style))
            elements.append(Spacer(1, 0.1*inch))
    

    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph("<i>Generated by VPPA Settlement Estimator</i>", normal_style))
    
    # Build PDF
    doc.build(elements)
    
    # Return buffer
    pdf_buffer.seek(0)
    return pdf_buffer


def _safe_pearson_correlation(series_a, series_b):
    """Robust Pearson correlation with graceful handling of sparse/constant data."""
    a = pd.to_numeric(pd.Series(series_a), errors="coerce")
    b = pd.to_numeric(pd.Series(series_b), errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return np.nan

    a = a[mask].astype(float)
    b = b[mask].astype(float)
    if np.isclose(a.std(ddof=0), 0.0) or np.isclose(b.std(ddof=0), 0.0):
        return 1.0 if np.allclose(a.values, b.values, atol=1e-12, rtol=0.0) else np.nan
    return float(a.corr(b))


def _compute_monthly_core_metrics(df_source, selected_month_numbers):
    """Compute monthly metrics used for multi-source correlation analysis."""
    if df_source is None or df_source.empty or "Time_Central" not in df_source.columns:
        return pd.DataFrame(columns=[
            "MonthPeriod", "Gen_MW", "Gen_Energy_MWh", "SPP", "Settlement_$",
            "Settlement_$/MWh", "Settlement_$_Uniform", "Settlement_$/MWh_Uniform",
            "Implied_REC_Cost_$"
        ])

    df = df_source.copy()
    df["Time_Central"] = pd.to_datetime(df["Time_Central"], errors="coerce")
    df = df.dropna(subset=["Time_Central"])
    if df.empty:
        return pd.DataFrame(columns=[
            "MonthPeriod", "Gen_MW", "Gen_Energy_MWh", "SPP", "Settlement_$",
            "Settlement_$/MWh", "Settlement_$_Uniform", "Settlement_$/MWh_Uniform",
            "Implied_REC_Cost_$"
        ])

    month_numbers = sorted(set(int(m) for m in (selected_month_numbers or [])))
    if month_numbers:
        df = df[df["Time_Central"].dt.month.isin(month_numbers)]
    if df.empty:
        return pd.DataFrame(columns=[
            "MonthPeriod", "Gen_MW", "Gen_Energy_MWh", "SPP", "Settlement_$",
            "Settlement_$/MWh", "Settlement_$_Uniform", "Settlement_$/MWh_Uniform",
            "Implied_REC_Cost_$"
        ])

    if "Gen_Energy_MWh" not in df.columns:
        df["Gen_Energy_MWh"] = 0.0
    if "Settlement_$" not in df.columns:
        df["Settlement_$"] = 0.0
    if "Settlement_$_Uniform" not in df.columns:
        df["Settlement_$_Uniform"] = 0.0
    if "SPP" not in df.columns:
        df["SPP"] = np.nan

    df["Gen_Energy_MWh"] = pd.to_numeric(df["Gen_Energy_MWh"], errors="coerce").fillna(0.0)
    df["Settlement_$"] = pd.to_numeric(df["Settlement_$"], errors="coerce").fillna(0.0)
    df["Settlement_$_Uniform"] = pd.to_numeric(df["Settlement_$_Uniform"], errors="coerce").fillna(0.0)
    df["SPP"] = pd.to_numeric(df["SPP"], errors="coerce")

    if "Gen_MW" in df.columns:
        df["Gen_MW"] = pd.to_numeric(df["Gen_MW"], errors="coerce")
    else:
        interval_hours = 0.25
        if len(df) >= 2:
            diffs = df["Time_Central"].sort_values().diff().dropna().dt.total_seconds() / 3600.0
            if not diffs.empty and diffs.median() > 0:
                interval_hours = float(diffs.median())
        df["Gen_MW"] = np.where(interval_hours > 0, df["Gen_Energy_MWh"] / interval_hours, np.nan)

    df["MonthPeriod"] = df["Time_Central"].dt.to_period("M")
    monthly = (
        df.groupby("MonthPeriod", as_index=False)
        .agg(
            {
                "Gen_MW": "mean",
                "Gen_Energy_MWh": "sum",
                "SPP": "mean",
                "Settlement_$": "sum",
                "Settlement_$_Uniform": "sum",
            }
        )
        .sort_values("MonthPeriod")
    )
    monthly["Settlement_$/MWh"] = np.where(
        monthly["Gen_Energy_MWh"] > 0,
        monthly["Settlement_$"] / monthly["Gen_Energy_MWh"],
        np.nan,
    )
    monthly["Settlement_$/MWh_Uniform"] = np.where(
        monthly["Gen_Energy_MWh"] > 0,
        monthly["Settlement_$_Uniform"] / monthly["Gen_Energy_MWh"],
        np.nan,
    )
    monthly["Implied_REC_Cost_$"] = np.where(
        monthly["Gen_Energy_MWh"] > 0,
        -(monthly["Settlement_$_Uniform"] / monthly["Gen_Energy_MWh"]),
        np.nan,
    )
    return monthly


def _compute_interval_core_metrics(df_source, selected_month_numbers):
    """Compute interval-level metrics used for per-month correlation analysis."""
    if df_source is None or df_source.empty or "Time_Central" not in df_source.columns:
        return pd.DataFrame(columns=[
            "Time_Central", "MonthPeriod", "Gen_MW", "Gen_Energy_MWh", "SPP",
            "Settlement_$", "Settlement_$/MWh", "Settlement_$_Uniform",
            "Settlement_$/MWh_Uniform", "Implied_REC_Cost_$"
        ])

    df = df_source.copy()
    df["Time_Central"] = pd.to_datetime(df["Time_Central"], errors="coerce")
    df = df.dropna(subset=["Time_Central"])
    if df.empty:
        return pd.DataFrame(columns=[
            "Time_Central", "MonthPeriod", "Gen_MW", "Gen_Energy_MWh", "SPP",
            "Settlement_$", "Settlement_$/MWh", "Settlement_$_Uniform",
            "Settlement_$/MWh_Uniform", "Implied_REC_Cost_$"
        ])

    month_numbers = sorted(set(int(m) for m in (selected_month_numbers or [])))
    if month_numbers:
        df = df[df["Time_Central"].dt.month.isin(month_numbers)]
    if df.empty:
        return pd.DataFrame(columns=[
            "Time_Central", "MonthPeriod", "Gen_MW", "Gen_Energy_MWh", "SPP",
            "Settlement_$", "Settlement_$/MWh", "Settlement_$_Uniform",
            "Settlement_$/MWh_Uniform", "Implied_REC_Cost_$"
        ])

    if "Gen_Energy_MWh" not in df.columns:
        df["Gen_Energy_MWh"] = 0.0
    if "Settlement_$" not in df.columns:
        df["Settlement_$"] = 0.0
    if "Settlement_$_Uniform" not in df.columns:
        df["Settlement_$_Uniform"] = 0.0
    if "SPP" not in df.columns:
        df["SPP"] = np.nan

    df["Gen_Energy_MWh"] = pd.to_numeric(df["Gen_Energy_MWh"], errors="coerce").fillna(0.0)
    df["Settlement_$"] = pd.to_numeric(df["Settlement_$"], errors="coerce").fillna(0.0)
    df["Settlement_$_Uniform"] = pd.to_numeric(df["Settlement_$_Uniform"], errors="coerce").fillna(0.0)
    df["SPP"] = pd.to_numeric(df["SPP"], errors="coerce")

    if "Gen_MW" in df.columns:
        df["Gen_MW"] = pd.to_numeric(df["Gen_MW"], errors="coerce")
    else:
        interval_hours = 0.25
        if len(df) >= 2:
            diffs = df["Time_Central"].sort_values().diff().dropna().dt.total_seconds() / 3600.0
            if not diffs.empty and diffs.median() > 0:
                interval_hours = float(diffs.median())
        df["Gen_MW"] = np.where(interval_hours > 0, df["Gen_Energy_MWh"] / interval_hours, np.nan)

    df["Settlement_$/MWh"] = np.where(
        df["Gen_Energy_MWh"] > 0,
        df["Settlement_$"] / df["Gen_Energy_MWh"],
        np.nan,
    )
    df["Settlement_$/MWh_Uniform"] = np.where(
        df["Gen_Energy_MWh"] > 0,
        df["Settlement_$_Uniform"] / df["Gen_Energy_MWh"],
        np.nan,
    )
    df["Implied_REC_Cost_$"] = np.where(
        df["Gen_Energy_MWh"] > 0,
        -(df["Settlement_$_Uniform"] / df["Gen_Energy_MWh"]),
        np.nan,
    )
    df["MonthPeriod"] = df["Time_Central"].dt.to_period("M")
    df = df.sort_values("Time_Central").drop_duplicates(subset=["Time_Central"], keep="last")

    return df[[
        "Time_Central", "MonthPeriod", "Gen_MW", "Gen_Energy_MWh", "SPP",
        "Settlement_$", "Settlement_$/MWh", "Settlement_$_Uniform",
        "Settlement_$/MWh_Uniform", "Implied_REC_Cost_$"
    ]]


def build_multi_source_correlation_analysis(preview_results, selected_month_numbers):
    """
    Build pairwise interval-level Pearson correlations for SCED/Model/Invoice metrics
    across all selected months (same method as by-month tables, but over the full period).
    Returns DataFrame indexed by metric with one column per available source pair.
    """
    interval_by_label = {}
    source_candidates = [
        ("SCED_Actual", "SCED Actual"),
        ("Model", "Model"),
        ("Settlement_Invoice", "Invoice"),
        ("Invoice", "Invoice"),
    ]
    for source_key, source_label in source_candidates:
        if source_label in interval_by_label:
            continue
        if source_key not in preview_results:
            continue
        interval_df = _compute_interval_core_metrics(preview_results.get(source_key), selected_month_numbers)
        if not interval_df.empty:
            interval_by_label[source_label] = interval_df

    pair_defs = [
        ("SCED Actual vs Model", "SCED Actual", "Model"),
        ("SCED Actual vs Invoice", "SCED Actual", "Invoice"),
        ("Model vs Invoice", "Model", "Invoice"),
    ]
    available_pairs = [(name, left, right) for name, left, right in pair_defs if left in interval_by_label and right in interval_by_label]
    if not available_pairs:
        return pd.DataFrame()

    metrics = [
        ("Gen_MW", "Gen MW"),
        ("Gen_Energy_MWh", "Energy MWh"),
        ("SPP", "SPP ($/MWh)"),
        # Use uniform (raw hub SPP) settlement columns so all three sources
        # are on the same price basis and the Pearson R is meaningful.
        ("Settlement_$_Uniform", "Settlement $ (Hub)"),
        ("Settlement_$/MWh_Uniform", "Settlement $/MWh (Hub)"),
        ("Implied_REC_Cost_$", "Implied REC Cost $"),
    ]

    out = pd.DataFrame({"Metric": [label for _, label in metrics]})
    for pair_name, left_label, right_label in available_pairs:
        left_df = interval_by_label[left_label][["Time_Central"] + [key for key, _ in metrics]]
        right_df = interval_by_label[right_label][["Time_Central"] + [key for key, _ in metrics]]
        merged = left_df.merge(right_df, on="Time_Central", how="inner", suffixes=("_left", "_right"))

        pair_vals = []
        for metric_key, _ in metrics:
            if merged.empty:
                pair_vals.append(np.nan)
                continue
            pair_vals.append(
                _safe_pearson_correlation(
                    merged[f"{metric_key}_left"],
                    merged[f"{metric_key}_right"],
                )
            )
        out[pair_name] = pair_vals

    return out.set_index("Metric")


def build_multi_source_correlation_by_month(preview_results, selected_month_numbers):
    """
    Build pairwise interval-level Pearson correlations grouped by calendar month.
    Returns dict: {"Jan 2025": DataFrame(metric x pair), ...}
    """
    interval_by_label = {}
    source_candidates = [
        ("SCED_Actual", "SCED Actual"),
        ("Model", "Model"),
        ("Settlement_Invoice", "Invoice"),
        ("Invoice", "Invoice"),
    ]
    for source_key, source_label in source_candidates:
        if source_label in interval_by_label:
            continue
        if source_key not in preview_results:
            continue
        interval_df = _compute_interval_core_metrics(preview_results.get(source_key), selected_month_numbers)
        if not interval_df.empty:
            interval_by_label[source_label] = interval_df

    pair_defs = [
        ("SCED Actual vs Model", "SCED Actual", "Model"),
        ("SCED Actual vs Invoice", "SCED Actual", "Invoice"),
        ("Model vs Invoice", "Model", "Invoice"),
    ]
    available_pairs = [(name, left, right) for name, left, right in pair_defs if left in interval_by_label and right in interval_by_label]
    if not available_pairs:
        return {}

    metrics = [
        ("Gen_MW", "Gen MW"),
        ("Gen_Energy_MWh", "Energy MWh"),
        ("SPP", "SPP ($/MWh)"),
        # Use uniform (raw hub SPP) settlement columns so all three sources
        # are on the same price basis and the Pearson R is meaningful.
        ("Settlement_$_Uniform", "Settlement $ (Hub)"),
        ("Settlement_$/MWh_Uniform", "Settlement $/MWh (Hub)"),
        ("Implied_REC_Cost_$", "Implied REC Cost $"),
    ]

    month_periods = sorted(
        set(
            p
            for df in interval_by_label.values()
            for p in df["MonthPeriod"].dropna().unique().tolist()
        )
    )
    out = {}
    for month_period in month_periods:
        month_df = pd.DataFrame({"Metric": [label for _, label in metrics]})
        for pair_name, left_label, right_label in available_pairs:
            left_df = interval_by_label[left_label]
            left_df = left_df[left_df["MonthPeriod"] == month_period][["Time_Central"] + [k for k, _ in metrics]]
            right_df = interval_by_label[right_label]
            right_df = right_df[right_df["MonthPeriod"] == month_period][["Time_Central"] + [k for k, _ in metrics]]
            merged = left_df.merge(right_df, on="Time_Central", how="inner", suffixes=("_left", "_right"))

            pair_vals = []
            for metric_key, _ in metrics:
                if merged.empty:
                    pair_vals.append(np.nan)
                    continue
                pair_vals.append(
                    _safe_pearson_correlation(
                        merged[f"{metric_key}_left"],
                        merged[f"{metric_key}_right"],
                    )
                )
            month_df[pair_name] = pair_vals

        month_df = month_df.set_index("Metric")
        if month_df.notna().any().any():
            out[month_period.strftime("%b %Y")] = month_df

    return out


def build_monthly_comparison_report_excel(
    preview_results,
    selected_month_numbers,
    val_year,
    selected_project_name=None,
    selected_resource_id=None,
    report_context_label=None,
):
    """Build a formatted monthly comparison workbook for currently selected sources/months."""
    month_numbers = sorted(set(int(m) for m in (selected_month_numbers or [])))
    if not month_numbers:
        month_numbers = list(range(1, 13))

    preferred_order = ["SCED_Actual", "Model", "Invoice", "Settlement_Invoice", "Actual", "TMY", "P50"]
    source_keys = [k for k in preferred_order if k in preview_results]
    source_keys += [k for k in preview_results.keys() if k not in source_keys]
    if not source_keys:
        raise ValueError("No preview sources available for export.")

    source_label_map = {
        "SCED_Actual": "SCED Actual",
        "Model": "Model",
        "Invoice": "Invoice",
        "Settlement_Invoice": "Invoice",
        "Actual": "Actual",
        "TMY": "TMY",
        "P50": "P50",
    }
    source_color_map = {
        "SCED_Actual": "#E7D3C6",   # peach
        "Model": "#CF95CB",         # violet
        "Invoice": "#91CF5A",       # green
        "Settlement_Invoice": "#91CF5A",
        "Actual": "#B9DDEB",        # blue-grey
        "TMY": "#D9D9D9",           # neutral
        "P50": "#F4CFA4",           # sand
    }
    source_path_map = {
        "SCED_Actual": (
            f"sced_cache/{selected_resource_id}_{val_year}_full.parquet"
            if selected_resource_id else f"sced_cache/*_{val_year}.parquet"
        ),
        "Model": "Modeled profile + ERCOT RTM hub prices",
        "Invoice": "data_static/Settlement_Invoice_Actuals.parquet",
        "Settlement_Invoice": "data_static/Settlement_Invoice_Actuals.parquet",
        "Actual": "Actual weather profile + ERCOT RTM hub prices",
        "TMY": "TMY weather profile + ERCOT RTM hub prices",
        "P50": "Historical P50 profile + ERCOT RTM hub prices",
    }

    monthly_by_source = {}
    for source_key in source_keys:
        df = preview_results.get(source_key, pd.DataFrame()).copy()
        if df.empty or "Time_Central" not in df.columns:
            monthly_by_source[source_key] = pd.DataFrame(columns=["MonthPeriod", "Settlement_$", "Gen_Energy_MWh"])
            continue
        if "Settlement_$" not in df.columns:
            df["Settlement_$"] = 0.0
        if "Gen_Energy_MWh" not in df.columns:
            df["Gen_Energy_MWh"] = 0.0
        df["MonthPeriod"] = pd.to_datetime(df["Time_Central"], errors="coerce").dt.to_period("M")
        df = df.dropna(subset=["MonthPeriod"])
        if not df.empty:
            df = df[df["MonthPeriod"].dt.month.isin(month_numbers)]
        monthly = (
            df.groupby("MonthPeriod", as_index=False)[["Settlement_$", "Gen_Energy_MWh"]]
            .sum()
            .sort_values("MonthPeriod")
        )
        monthly_by_source[source_key] = monthly

    # Build row month periods from selected months in target year.
    selected_periods = [pd.Period(f"{int(val_year)}-{m:02d}", freq="M") for m in month_numbers]
    has_any = False
    for source_key in source_keys:
        monthly = monthly_by_source[source_key]
        if monthly.empty:
            continue
        if set(monthly["MonthPeriod"]).intersection(selected_periods):
            has_any = True
            break
    row_periods = selected_periods if has_any else sorted(
        set(
            p for source_key in source_keys
            for p in monthly_by_source[source_key].get("MonthPeriod", pd.Series(dtype=object)).tolist()
        )
    )

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        sheet_name = "Monthly Report"
        worksheet = workbook.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = worksheet

        title_fmt = workbook.add_format({
            "bold": True, "font_size": 14, "align": "left", "valign": "vcenter"
        })
        subtitle_fmt = workbook.add_format({
            "italic": True, "font_color": "#555555", "align": "left"
        })
        row_label_header_fmt = workbook.add_format({
            "bold": True, "bg_color": "#B7DCE9", "border": 1, "align": "center", "valign": "vcenter"
        })
        row_label_fmt = workbook.add_format({"border": 1})
        row_label_total_fmt = workbook.add_format({
            "bold": True, "bg_color": "#B7DCE9", "border": 1
        })
        money_fmt = workbook.add_format({"num_format": "$#,##0", "border": 1})
        energy_fmt = workbook.add_format({"num_format": "#,##0.0", "border": 1})
        money_total_fmt = workbook.add_format({
            "num_format": "$#,##0", "border": 1, "bold": True, "bg_color": "#EFEFEF"
        })
        energy_total_fmt = workbook.add_format({
            "num_format": "#,##0.0", "border": 1, "bold": True, "bg_color": "#EFEFEF"
        })
        group_header_fmts = {}
        metric_header_fmts = {}
        for source_key in source_keys:
            src_color = source_color_map.get(source_key, "#D9D9D9")
            group_header_fmts[source_key] = workbook.add_format({
                "bold": True, "bg_color": src_color, "border": 1, "align": "center", "valign": "vcenter"
            })
            metric_header_fmts[source_key] = workbook.add_format({
                "bold": True, "bg_color": src_color, "border": 1, "text_wrap": True, "align": "center", "valign": "vcenter"
            })

        worksheet.write(0, 0, "Monthly Comparison Report", title_fmt)
        source_title = report_context_label or selected_project_name or "Selected Source"
        worksheet.write(1, 0, f"Project: {source_title} | Year: {val_year} | Months: {', '.join(pd.Timestamp(1900, m, 1).strftime('%B') for m in month_numbers)}", subtitle_fmt)

        start_row = 3
        worksheet.merge_range(start_row, 0, start_row + 1, 0, "Row Labels", row_label_header_fmt)

        col = 1
        for source_key in source_keys:
            source_label = source_label_map.get(source_key, source_key)
            clean_label = source_label.replace(" ", "_")
            worksheet.merge_range(start_row, col, start_row, col + 1, f"{source_label}", group_header_fmts[source_key])
            worksheet.write(start_row + 1, col, f"Sum of Settlement_$_{clean_label}", metric_header_fmts[source_key])
            worksheet.write(start_row + 1, col + 1, f"Sum of Gen_Energy_MWh_{clean_label}", metric_header_fmts[source_key])
            col += 2

        data_start = start_row + 2
        current_row = data_start
        for month_period in row_periods:
            month_label = month_period.strftime("%B")
            worksheet.write(current_row, 0, month_label, row_label_fmt)
            col = 1
            for source_key in source_keys:
                monthly = monthly_by_source[source_key]
                settlement = 0.0
                energy = 0.0
                if not monthly.empty:
                    hit = monthly[monthly["MonthPeriod"] == month_period]
                    if not hit.empty:
                        settlement = float(hit["Settlement_$"].iloc[0])
                        energy = float(hit["Gen_Energy_MWh"].iloc[0])
                worksheet.write_number(current_row, col, settlement, money_fmt)
                worksheet.write_number(current_row, col + 1, energy, energy_fmt)
                col += 2
            current_row += 1

        total_row = current_row
        worksheet.write(total_row, 0, "Grand Total", row_label_total_fmt)
        col = 1
        for source_key in source_keys:
            first_row = data_start + 1
            last_row = total_row
            settlement_col = col
            energy_col = col + 1
            settlement_col_letter = chr(ord("A") + settlement_col)
            energy_col_letter = chr(ord("A") + energy_col)
            worksheet.write_formula(
                total_row,
                settlement_col,
                f"=SUM({settlement_col_letter}{first_row}:{settlement_col_letter}{last_row})",
                money_total_fmt,
            )
            worksheet.write_formula(
                total_row,
                energy_col,
                f"=SUM({energy_col_letter}{first_row}:{energy_col_letter}{last_row})",
                energy_total_fmt,
            )
            col += 2

        # Multi-source correlation section (monthly pairwise Pearson R)
        corr_df = build_multi_source_correlation_analysis(preview_results, month_numbers)
        if not corr_df.empty:
            corr_start = total_row + 3
            corr_end_col = len(corr_df.columns)
            corr_title_fmt = workbook.add_format({
                "bold": True, "font_color": "white", "bg_color": "#1F3A68", "border": 1, "align": "center"
            })
            corr_metric_header_fmt = workbook.add_format({
                "bold": True, "font_color": "white", "bg_color": "#1F3A68", "border": 1, "align": "center"
            })
            pair_header_colors = {
                "SCED Actual vs Model": "#6A1B9A",
                "SCED Actual vs Invoice": "#F57C00",
                "Model vs Invoice": "#0F9D58",
            }

            metric_cell_fmt = workbook.add_format({"border": 1})
            na_cell_fmt = workbook.add_format({"border": 1, "align": "center"})
            corr_good_fmt = workbook.add_format({
                "num_format": "0.0000", "bg_color": "#0F9D58", "font_color": "white", "border": 1, "align": "center"
            })
            corr_ok_fmt = workbook.add_format({
                "num_format": "0.0000", "bg_color": "#F6BF26", "font_color": "#1F1F1F", "border": 1, "align": "center"
            })
            corr_bad_fmt = workbook.add_format({
                "num_format": "0.0000", "bg_color": "#D93025", "font_color": "white", "border": 1, "align": "center"
            })
            corr_good_threshold = 0.80
            corr_ok_threshold = 0.60

            def _corr_quality_format(val):
                if pd.isna(val):
                    return na_cell_fmt
                if float(val) >= corr_good_threshold:
                    return corr_good_fmt
                if float(val) >= corr_ok_threshold:
                    return corr_ok_fmt
                return corr_bad_fmt

            worksheet.merge_range(
                corr_start,
                0,
                corr_start,
                corr_end_col,
                f"{source_title} {val_year} - Multi-Source Correlation Analysis",
                corr_title_fmt,
            )
            corr_header_row = corr_start + 1
            worksheet.write(corr_header_row, 0, "Metric", corr_metric_header_fmt)
            for idx, pair_name in enumerate(corr_df.columns, start=1):
                pair_color = pair_header_colors.get(pair_name, "#2E75B6")
                pair_header_fmt = workbook.add_format({
                    "bold": True,
                    "font_color": "white",
                    "bg_color": pair_color,
                    "border": 1,
                    "align": "center",
                })
                worksheet.write(corr_header_row, idx, pair_name, pair_header_fmt)

            corr_data_row = corr_header_row + 1
            for metric_name, row_vals in corr_df.iterrows():
                worksheet.write(corr_data_row, 0, metric_name, metric_cell_fmt)
                for idx, pair_name in enumerate(corr_df.columns, start=1):
                    val = row_vals.get(pair_name)
                    if pd.notna(val):
                        worksheet.write_number(corr_data_row, idx, float(val), _corr_quality_format(val))
                    else:
                        worksheet.write(corr_data_row, idx, "-", na_cell_fmt)
                corr_data_row += 1

            key_row = corr_data_row + 1
            key_label_fmt = workbook.add_format({"italic": True, "font_color": "#444444"})
            worksheet.write(key_row, 0, "Source color key:", key_label_fmt)
            for idx, pair_name in enumerate(corr_df.columns, start=1):
                pair_color = pair_header_colors.get(pair_name, "#2E75B6")
                key_fmt = workbook.add_format({
                    "bg_color": pair_color,
                    "font_color": "white",
                    "border": 1,
                    "align": "center",
                    "italic": True,
                })
                worksheet.write(key_row, idx, pair_name, key_fmt)

            quality_key_row = key_row + 1
            worksheet.write(quality_key_row, 0, "Correlation quality key:", key_label_fmt)
            quality_good_fmt = workbook.add_format({"bg_color": "#0F9D58", "font_color": "white", "border": 1, "align": "center", "italic": True})
            quality_ok_fmt = workbook.add_format({"bg_color": "#F6BF26", "font_color": "#1F1F1F", "border": 1, "align": "center", "italic": True})
            quality_bad_fmt = workbook.add_format({"bg_color": "#D93025", "font_color": "white", "border": 1, "align": "center", "italic": True})
            worksheet.write(quality_key_row, 1, f"Good >= {corr_good_threshold:.2f}", quality_good_fmt)
            worksheet.write(quality_key_row, 2, f"OK {corr_ok_threshold:.2f}-{corr_good_threshold - 0.01:.2f}", quality_ok_fmt)
            worksheet.write(quality_key_row, 3, f"Bad < {corr_ok_threshold:.2f}", quality_bad_fmt)

            corr_month_row = quality_key_row + 2
            corr_by_month = build_multi_source_correlation_by_month(preview_results, month_numbers)
            if corr_by_month:
                month_section_fmt = workbook.add_format({
                    "bold": True, "font_color": "white", "bg_color": "#2E75B6", "border": 1, "align": "center"
                })
                for month_label, month_corr_df in corr_by_month.items():
                    worksheet.merge_range(
                        corr_month_row,
                        0,
                        corr_month_row,
                        corr_end_col,
                        f"{month_label} Correlation Analysis",
                        month_section_fmt,
                    )
                    month_header_row = corr_month_row + 1
                    worksheet.write(month_header_row, 0, "Metric", corr_metric_header_fmt)
                    for idx, pair_name in enumerate(month_corr_df.columns, start=1):
                        pair_color = pair_header_colors.get(pair_name, "#2E75B6")
                        pair_header_fmt = workbook.add_format({
                            "bold": True,
                            "font_color": "white",
                            "bg_color": pair_color,
                            "border": 1,
                            "align": "center",
                        })
                        worksheet.write(month_header_row, idx, pair_name, pair_header_fmt)

                    month_data_row = month_header_row + 1
                    for metric_name, row_vals in month_corr_df.iterrows():
                        worksheet.write(month_data_row, 0, metric_name, metric_cell_fmt)
                        for idx, pair_name in enumerate(month_corr_df.columns, start=1):
                            val = row_vals.get(pair_name)
                            if pd.notna(val):
                                worksheet.write_number(month_data_row, idx, float(val), _corr_quality_format(val))
                            else:
                                worksheet.write(month_data_row, idx, "-", na_cell_fmt)
                        month_data_row += 1
                    corr_month_row = month_data_row + 1

            def_start = corr_month_row + 1
        else:
            def_start = total_row + 3

        # Metric definitions section
        worksheet.merge_range(def_start, 0, def_start, 2, "Metric Definitions", workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#1F4E78", "border": 1, "align": "left"
        }))
        worksheet.write(def_start + 1, 0, "Metric", workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#2E75B6", "border": 1
        }))
        worksheet.write(def_start + 1, 1, "Data Source / Path", workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#2E75B6", "border": 1
        }))
        worksheet.write(def_start + 1, 2, "Description", workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#2E75B6", "border": 1
        }))

        def_row = def_start + 2
        for source_key in source_keys:
            source_label = source_label_map.get(source_key, source_key)
            src_color = source_color_map.get(source_key, "#D9D9D9")
            cell_fmt = workbook.add_format({"bg_color": src_color, "border": 1, "text_wrap": True, "valign": "top"})
            metric_1 = f"Settlement $ - {source_label}"
            metric_2 = f"Energy (MWh) - {source_label}"
            src_path = source_path_map.get(source_key, source_label)
            desc_settle = f"Total VPPA settlement dollars calculated from {source_label.lower()} for the selected months."
            desc_energy = f"Total settled energy (MWh) from {source_label.lower()} for the selected months."
            worksheet.write(def_row, 0, metric_1, cell_fmt)
            worksheet.write(def_row, 1, src_path, cell_fmt)
            worksheet.write(def_row, 2, desc_settle, cell_fmt)
            def_row += 1
            worksheet.write(def_row, 0, metric_2, cell_fmt)
            worksheet.write(def_row, 1, src_path, cell_fmt)
            worksheet.write(def_row, 2, desc_energy, cell_fmt)
            def_row += 1

            if source_key == "SCED_Actual":
                link_fmt = workbook.add_format({"font_color": "#0563C1", "underline": 1})
                worksheet.write_url(
                    def_row,
                    0,
                    "https://www.ercot.com/mp/data-products/data-product-details?id=NP3-965-ER",
                    link_fmt,
                    "https://www.ercot.com/mp/data-products/data-product-details?id=NP3-965-ER",
                )
                def_row += 1

        footnote_fmt = workbook.add_format({
            "italic": True, "font_color": "#666666", "text_wrap": True
        })
        worksheet.merge_range(
            def_row + 1,
            0,
            def_row + 1,
            2,
            "Note: Minor month-end variance can occur from interval boundary conventions (e.g., 00:00 treatment at month rollover).",
            footnote_fmt,
        )

        worksheet.set_column(0, 0, 20)
        for idx in range(len(source_keys)):
            base_col = 1 + (idx * 2)
            worksheet.set_column(base_col, base_col, 20)
            worksheet.set_column(base_col + 1, base_col + 1, 18)
        worksheet.freeze_panes(data_start, 1)

    output.seek(0)
    return output.getvalue()

# ...

# --- Sidebar: Scenario Builder ---
# --- Helper Functions for Sidebar ---
def reset_defaults():
    st.session_state.scenarios = []
    st.session_state.sb_techs = ["Solar"]
    # st.session_state.sb_select_all_years = False  # Removed
    st.session_state.sb_years = [2025]
    st.session_state.sb_hubs = ["HB_NORTH"]
    st.session_state.sb_use_specific_month = False
    st.session_state.sb_months = ["January"]
    st.session_state.sb_capacity = 80.0
    st.session_state.sb_vppa_price = 50.0
    st.session_state.sb_no_curtailment = False
    st.session_state.sb_force_tmy = False
    st.session_state.sb_wind_weather_source_label = "Open-Meteo / PVGIS (Default)"
    st.session_state.sb_wind_model_engine_label = "Standard (Current)"
    st.session_state.sb_hrrr_forecast_hour = 0


# --- Solar/Wind Batch Form ---
# --- Solar/Wind Batch Form ---
# Form Removed to allow dynamic updates (monthly filter)
with st.sidebar:
    st.header("Scenario Builder")
    
    # --- Generation Source ---
    # --- Generation Source ---
    sb_gen_source = st.selectbox(
        "Generation Source",
        ["Solar", "Wind", "ERCOT Asset", "Load (Future)", "Custom Upload"],
        key="sb_gen_source",
        help="Choose the type of generation profile."
    )
    
    # --- ERCOT Asset Logic ---
    selected_asset_info = None
    if sb_gen_source == "ERCOT Asset":
        assets_file = "ercot_assets.json"
        
        # Load Assets
        if os.path.exists(assets_file):
            with open(assets_file, "r") as f:
                ercot_assets = json.load(f)
            
            asset_names = sorted(list(ercot_assets.keys()))
            try:
                def_idx = asset_names.index("Azure Sky Wind")
            except ValueError:
                def_idx = 0
                
            sb_selected_asset = st.selectbox(
                "Select Asset",
                asset_names,
                index=def_idx,
                key="sb_selected_asset"
            )
            
            if sb_selected_asset:
                selected_asset_info = ercot_assets[sb_selected_asset]
                selected_resource_id = selected_asset_info.get("resource_name")
                first_sced_date = get_first_cached_sced_date(selected_resource_id)
                if first_sced_date:
                    st.caption(f"SCED first available: {first_sced_date}")
                elif selected_resource_id:
                    st.caption("SCED first available: not cached locally")

                # Keep Scenario Builder location synced to selected asset metadata.
                try:
                    asset_lat = float(selected_asset_info.get("lat"))
                    asset_lon = float(selected_asset_info.get("lon"))
                    st.session_state.sb_custom_lat = asset_lat
                    st.session_state.sb_custom_lon = asset_lon
                    st.session_state.map_lat = asset_lat
                    st.session_state.map_lon = asset_lon
                except (TypeError, ValueError):
                    pass
                
                # Auto-set Tech (Update session state for multiselect)
                tech = selected_asset_info.get("tech", "Wind")
                st.session_state.sb_techs = [tech]
                
                # Azure Sky Specifics
                if sb_selected_asset == "Azure Sky Wind":
                    st.markdown("### Azure Sky Data Source")
                    sb_az_source = st.radio(
                        "Source Type",
                        ["Bill (Actuals)", "SCED (ERCOT)", "Modeled"],
                        horizontal=True,
                        key="sb_az_source"
                    )
                    
                    # Defaults
                    if 'sb_vppa_price' not in st.session_state or st.session_state.sb_vppa_price != 17.34:
                         st.session_state.sb_vppa_price = 17.34
                    
                    cap = selected_asset_info.get("capacity_mw", 350.0)
                    st.session_state.sb_capacity = float(cap)
                else:
                    # Generic
                    cap = selected_asset_info.get("capacity_mw", 100.0)
                    st.session_state.sb_capacity = float(cap)
                    
        else:
            st.error("ercot_assets.json not found!")
    
    # --- Years/Hubs Selection ---
    available_years = [2026, 2025, 2024, 2023, 2022, 2021, 2020]
    common_hubs = ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON", "HB_PAN"]
    
    s_techs = st.multiselect("Generation Source", ["Solar", "Wind"], default=["Solar"], key="sb_techs")
    if not s_techs:
        st.warning("Please select at least one technology.")

    st.markdown("*Select multiple years/hubs*")
    
    # Handle "Select All" logic for Years
    if "sb_years" not in st.session_state:
        st.session_state.sb_years = [2025]
        
    if "Select All" in st.session_state.sb_years:
        st.session_state.sb_years = available_years
        st.rerun()

    s_years = st.multiselect("Years", ["Select All"] + available_years, key="sb_years")
    if not s_years:
        st.warning("Please select at least one year.")
    
    # --- Map Location Picker (Moved before Hub selection) ---
    with st.expander("🗺️ Pick Location on Map", expanded=False):
        st.caption("Search by name or click on the map")
        
        # Location search box
        search_query = st.text_input("🔍 Search location", placeholder="e.g., Abilene, TX or 79601", key="location_search")
        
        if search_query:
            try:
                geolocator = Nominatim(user_agent="vppa_estimator")
                # Append Texas to improve search accuracy
                if "texas" not in search_query.lower() and "tx" not in search_query.lower():
                    search_with_state = f"{search_query}, Texas, USA"
                else:
                    search_with_state = f"{search_query}, USA"
                
                location = geolocator.geocode(search_with_state, timeout=5)
                
                if location:
                    # Clamp to Texas bounds
                    found_lat = max(25.5, min(36.5, location.latitude))
                    found_lon = max(-106.5, min(-93.5, location.longitude))
                    
                    st.session_state.map_lat = found_lat
                    st.session_state.map_lon = found_lon
                    st.session_state.sb_custom_lat = found_lat
                    st.session_state.sb_custom_lon = found_lon
                    # Auto-check the "Use Custom Location" checkbox
                    st.session_state.sb_use_custom_location = True
                    st.success(f"📍 Found: {location.address[:50]}...")
                    st.caption(f"Coordinates: {found_lat:.4f}, {found_lon:.4f}")
                else:
                    st.warning("Location not found. Try a different name.")
            except GeocoderTimedOut:
                st.warning("Search timed out. Try again.")
            except Exception as e:
                st.error(f"Search error: {str(e)[:50]}")
        
        
        # Initialize map location from session state or defaults
        if 'map_lat' not in st.session_state:
            st.session_state.map_lat = 32.0
        if 'map_lon' not in st.session_state:
            st.session_state.map_lon = -100.0
        
        # Sync map with custom location inputs if manually entered
        if 'sb_custom_lat' in st.session_state and 'sb_custom_lon' in st.session_state:
            st.session_state.map_lat = st.session_state.sb_custom_lat
            st.session_state.map_lon = st.session_state.sb_custom_lon
        
        # Create map centered on Texas/ERCOT region
        m = folium.Map(
            location=[31.0, -100.0],  # Center of Texas
            zoom_start=6,
            tiles="OpenStreetMap"
        )
        
        # Add marker for current selected location
        folium.Marker(
            [st.session_state.map_lat, st.session_state.map_lon],
            popup=f"Selected: {st.session_state.map_lat:.4f}, {st.session_state.map_lon:.4f}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Add ERCOT hub markers for reference
        hub_locations = {
            "HB_NORTH": (32.3865, -96.8475),
            "HB_SOUTH": (26.9070, -99.2715),
            "HB_WEST": (32.4518, -100.5371),
            "HB_HOUSTON": (29.3013, -94.7977),
            "HB_PAN": (35.2220, -101.8313),
        }
        for hub, (lat, lon) in hub_locations.items():
            folium.CircleMarker(
                [lat, lon],
                radius=8,
                popup=hub,
                color='blue',
                fill=True,
                fillOpacity=0.6
            ).add_to(m)
        
        # Display map and capture clicks
        # Increased width to match main area
        map_data = st_folium(m, height=400, width=None, returned_objects=["last_clicked"], key="scenario_map")
        
        if map_data and map_data.get("last_clicked"):
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]
            # Clamp to Texas bounds to prevent errors in form inputs
            clicked_lat = max(25.5, min(36.5, clicked_lat))
            clicked_lon = max(-106.5, min(-93.5, clicked_lon))
            st.session_state.map_lat = clicked_lat
            st.session_state.map_lon = clicked_lon
            # Also sync to form input keys so they update
            st.session_state.sb_custom_lat = clicked_lat
            st.session_state.sb_custom_lon = clicked_lon
            # Auto-check the "Use Custom Location" checkbox
            st.session_state.sb_use_custom_location = True
            
            # Calculate nearest hub on click and auto-select it
            def calc_dist(lat1, lon1, lat2, lon2):
                return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
            click_distances = {hub: calc_dist(clicked_lat, clicked_lon, lat, lon) for hub, (lat, lon) in hub_locations.items()}
            nearest = min(click_distances, key=click_distances.get)
            st.session_state.sb_hubs = [nearest]
            
            st.success(f"📍 Selected: {clicked_lat:.4f}, {clicked_lon:.4f}")
        
        # Calculate and suggest nearest hub (re-defined locally to ensure access)
        def calc_distance(lat1, lon1, lat2, lon2):
            """Simple Euclidean distance (good enough for nearby points)"""
            return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
        
        current_lat = st.session_state.get('map_lat', 32.0)
        current_lon = st.session_state.get('map_lon', -100.0)
        
        distances = {}
        for hub_name, (hub_lat, hub_lon) in hub_locations.items():
            distances[hub_name] = calc_distance(current_lat, current_lon, hub_lat, hub_lon)
        
        nearest_hub = min(distances, key=distances.get)
        nearest_dist_miles = distances[nearest_hub] * 69  # Rough lat/lon to miles
        
        st.info(f"💡 **Suggested Hub:** {nearest_hub} (~{nearest_dist_miles:.0f} mi)")
        
        # Store suggested hub in session state for form to use
        st.session_state.suggested_hub = nearest_hub
        
        st.caption("🔵 Blue = Hub locations | 🔴 Red = Your selection")

    # Use suggested hub from map if available, otherwise default to HB_NORTH
    default_hub = st.session_state.get('suggested_hub', 'HB_NORTH')
    if default_hub not in common_hubs:
        default_hub = 'HB_NORTH'
    
    s_hubs = st.multiselect("Hubs", common_hubs, default=[default_hub], key="sb_hubs")
    if not s_hubs:
        st.warning("Please select at least one hub.")
    
    # Duration Selection
    use_specific_month = st.checkbox("Filter by specific month", key="sb_use_specific_month")
    s_duration = "Specific Month" if use_specific_month else "Full Year"
    
    s_months = None
    if use_specific_month:
        all_months = [
            "January", "February", "March", "April", "May", "June", 
            "July", "August", "September", "October", "November", "December"
        ]
        s_months = st.multiselect("Months", all_months, default=["January"], key="sb_months")
        if not s_months:
            st.warning("Please select at least one month.")
    
    s_capacity = st.number_input("Capacity (MW)", value=80.0, step=10.0, key="sb_capacity")
    s_vppa_price = st.number_input("VPPA Price ($/MWh)", value=50.0, step=1.0, key="sb_vppa_price")
    
    # Revenue Share Option (configurable upside split)
    s_revenue_share_pct = st.number_input(
        "Buyer's Upside Share % (when SPP > PPA)", 
        min_value=0, 
        max_value=100, 
        value=100, 
        step=5,
        help="% of upside buyer receives when SPP > PPA price. 100% = standard PPA (buyer keeps all upside). 50% = 50/50 split with seller.",
        key="sb_revenue_share_pct"
    )
    
    # Curtailment Option
    s_no_curtailment = st.checkbox("Remove $0 floor (No Curtailment)", key="sb_no_curtailment")

    # TMY Override
    s_force_tmy = st.checkbox("Force TMY Data (Override Actuals)", value=False, help="Use typical weather data.", key="sb_force_tmy")

    s_wind_weather_source = "AUTO"
    s_hrrr_forecast_hour = 0
    s_wind_model_engine = "STANDARD"
    if "Wind" in s_techs:
        w_col1, w_col2, w_col3 = st.columns([2, 1, 2])
        with w_col1:
            source_labels = list(WIND_WEATHER_SOURCE_OPTIONS.keys())
            selected_label = st.selectbox(
                "Wind Weather Dataset",
                source_labels,
                key="sb_wind_weather_source_label",
                help="NOAA HRRR uses cached files under data_cache/hrrr (generated separately).",
            )
            s_wind_weather_source = WIND_WEATHER_SOURCE_OPTIONS.get(selected_label, "AUTO")
        with w_col2:
            if s_wind_weather_source == "NOAA_HRRR_CACHED":
                s_hrrr_forecast_hour = int(
                    st.number_input(
                        "HRRR Forecast Hour (fxx)",
                        min_value=0,
                        max_value=18,
                        value=int(st.session_state.get("sb_hrrr_forecast_hour", 0)),
                        step=1,
                        key="sb_hrrr_forecast_hour",
                        help="0 = analysis/nowcast. 1..18 = lead forecast hour.",
                    )
                )
                st.caption("Using cached NOAA HRRR from `data_cache/hrrr` when available.")
                hrrr_count = get_hrrr_cache_count()
                if hrrr_count == 0:
                    st.warning("No HRRR cache files found yet. Generate them with `scripts/fetch_hrrr_wind.py`.")
                else:
                    st.caption(f"Detected {hrrr_count} cached HRRR files.")
        with w_col3:
            model_labels = list(WIND_MODEL_ENGINE_OPTIONS.keys())
            model_label = st.selectbox(
                "Wind Model Engine",
                model_labels,
                index=0,
                key="sb_wind_model_engine_label",
                help="Advanced mode applies monthly EIA/CF targets, SCED bias correction, node adjustments, and tuned clipping.",
            )
            s_wind_model_engine = WIND_MODEL_ENGINE_OPTIONS.get(model_label, "STANDARD")
    

    st.markdown("---")
    
    # Two buttons: Add (append) vs Clear & Run (reset)
    # Three buttons: Add (append), Clear & Run (replace), Reset All (clear)
    # Button Layout: 
    # Row 1: Add (Primary Action)
    # --- Map Location Picker (Moved from sidebar) ---


    # Custom Location Override (Moved after Map)
    s_use_custom_location = st.checkbox("Use Custom Project Location", value=False, help="Enter coordinates to override hub defaults.", key="sb_use_custom_location")
    
    s_custom_lat = None
    s_custom_lon = None
    if s_use_custom_location:
        st.caption("💡 Enter your project's coordinates (or use map picker above)")
        col_lat, col_lon = st.columns(2)
        with col_lat:
            s_custom_lat = st.number_input("Latitude", min_value=25.0, max_value=40.0, value=32.0, step=0.01, format="%.4f", key="sb_custom_lat")
        with col_lon:
            s_custom_lon = st.number_input("Longitude", min_value=-107.0, max_value=-93.0, value=-100.0, step=0.01, format="%.4f", key="sb_custom_lon")

    # Row 1: Add (Primary Action)
    add_button = st.button("➕ Add Scenarios", type="primary", use_container_width=True)
    
    # Row 2: Secondary Actions
    col_clear, col_reset = st.columns(2)
    with col_clear:
        # Run button triggers calculation
        clear_run_button = st.button("🏃 Run", type="secondary", use_container_width=True)
    with col_reset:
        reset_all_button = st.button("🗑️ Reset", type="secondary", use_container_width=True, on_click=reset_defaults)
    
    # Handle Add Scenarios (append mode)
    if add_button:
        if not s_years or not s_hubs or not s_techs or (use_specific_month and not s_months):
            st.error("Please ensure Years, Hubs, Types, and Months (if applicable) are selected.")
        else:
            # Helper for friendly names
            hub_map = {
                "HB_NORTH": "North Hub", "HB_SOUTH": "South Hub", "HB_WEST": "West Hub", "HB_HOUSTON": "Houston Hub"
            }
            
            added_count = 0
            
            # Iterate through all combinations
            for year in s_years:
                for hub in s_hubs:
                    for tech in s_techs:
                        friendly_hub = hub_map.get(hub, hub)
                        
                        # Define list of monthly iterations
                        month_iterator = s_months if use_specific_month else [None]
                        
                        for month in month_iterator:
                            # Construct Name
                            if use_specific_month:
                                name = f"{month} {year} {tech} in {friendly_hub} ({int(s_capacity)}MW)"
                            else:
                                name = f"{year} {tech} in {friendly_hub} ({int(s_capacity)}MW)"
                            
                            if s_no_curtailment:
                                name += " [No Curtailment]"
                            
                            if s_revenue_share_pct < 100:
                                name += f" [{s_revenue_share_pct}% Share]"
                            
                            if s_force_tmy:
                                name += " [TMY]"
                            if tech == "Wind" and s_wind_weather_source == "NOAA_HRRR_CACHED":
                                name += f" [HRRR f{int(s_hrrr_forecast_hour):02d}]"
                            if tech == "Wind" and s_wind_model_engine == "ADVANCED_CALIBRATED":
                                name += " [Advanced Wind]"
                            
                            if s_use_custom_location and s_custom_lat is not None:
                                name += f" [Custom: {s_custom_lat:.2f}, {s_custom_lon:.2f}]"
                                
                            # Check for duplicates
                            if any(s['name'] == name for s in st.session_state.scenarios):
                                continue 
                            else:
                                # Determine Source Type and Metadata
                                src_type = "MODEL"
                                res_id = None
                                ast_name = None
                                ast_cap = None
                                bill_path = None
                                
                                if sb_gen_source == "ERCOT Asset":
                                    src_type = "SCED" # Default
                                    ast_name = sb_selected_asset
                                    if selected_asset_info:
                                        res_id = selected_asset_info.get("resource_name")
                                        ast_cap = selected_asset_info.get("capacity_mw")
                                    
                                    if sb_selected_asset == "Azure Sky Wind":
                                        if sb_az_source == "Bill (Actuals)":
                                            src_type = "BILL"
                                            bill_path = "AzureSkyActuals.xlsx"
                                        elif sb_az_source == "Modeled":
                                            src_type = "MODEL"
                                        else:
                                            src_type = "SCED"

                                new_scenario = {
                                    "id": datetime.now().isoformat() + f"_{added_count}",
                                    "name": name,
                                    "year": year,
                                    "hub": hub,
                                    "tech": tech,
                                    "duration": s_duration,
                                    "month": month,
                                    "capacity_mw": s_capacity,
                                    "vppa_price": s_vppa_price,
                                    "no_curtailment": s_no_curtailment,
                                    "revenue_share_pct": s_revenue_share_pct,
                                    "force_tmy": s_force_tmy,
                                    "wind_weather_source": s_wind_weather_source if tech == "Wind" else "AUTO",
                                    "hrrr_forecast_hour": int(s_hrrr_forecast_hour) if tech == "Wind" else 0,
                                    "wind_model_engine": s_wind_model_engine if tech == "Wind" else "STANDARD",
                                    "custom_lat": s_custom_lat if s_use_custom_location else None,
                                    "custom_lon": s_custom_lon if s_use_custom_location else None,
                                    "custom_profile_path": None,
                                    "source_type": src_type,
                                    "resource_id": res_id,
                                    "asset_name": ast_name,
                                    "asset_capacity_mw": ast_cap,
                                    "bill_file_path": bill_path,
                                    "negative_price_floor": st.session_state.get("val_price_floor", -3.0) if st.session_state.get("val_use_price_floor", False) else None,
                            }
                            st.session_state.scenarios.append(new_scenario)
                            added_count += 1
            
            if added_count > 0:
                st.success(f"Added {added_count} scenarios!")
                st.rerun()
            else:
                st.warning("No new scenarios added (duplicates or empty selection).")
    
    # Handle Clear & Run (reset mode)
    if clear_run_button:
        # Clear existing scenarios FIRST
        st.session_state.scenarios = []
        
        if not s_years or not s_hubs or not s_techs or (use_specific_month and not s_months):
            st.error("Please ensure Years, Hubs, Types, and Months (if applicable) are selected.")
        else:
            # Helper for friendly names
            hub_map = {
                "HB_NORTH": "North Hub", "HB_SOUTH": "South Hub", "HB_WEST": "West Hub", "HB_HOUSTON": "Houston Hub"
            }
            
            added_count = 0
            
            # Iterate through all combinations
            for year in s_years:
                for hub in s_hubs:
                    for tech in s_techs:
                        friendly_hub = hub_map.get(hub, hub)
                        
                        # Define list of monthly iterations
                        month_iterator = s_months if use_specific_month else [None]
                        
                        for month in month_iterator:
                            # Construct Name
                            if use_specific_month:
                                name = f"{month} {year} {tech} in {friendly_hub} ({int(s_capacity)}MW)"
                            else:
                                name = f"{year} {tech} in {friendly_hub} ({int(s_capacity)}MW)"
                            
                            if s_no_curtailment:
                                name += " [No Curtailment]"
                            
                            if s_revenue_share_pct < 100:
                                name += f" [{s_revenue_share_pct}% Share]"
                            
                            if s_force_tmy:
                                name += " [TMY]"
                            if tech == "Wind" and s_wind_weather_source == "NOAA_HRRR_CACHED":
                                name += f" [HRRR f{int(s_hrrr_forecast_hour):02d}]"
                            if tech == "Wind" and s_wind_model_engine == "ADVANCED_CALIBRATED":
                                name += " [Advanced Wind]"
                            
                            if s_use_custom_location and s_custom_lat is not None:
                                name += f" [Custom: {s_custom_lat:.2f}, {s_custom_lon:.2f}]"
                                
                            # Determine Source Type and Metadata
                            src_type = "MODEL"
                            res_id = None
                            ast_name = None
                            ast_cap = None
                            bill_path = None
                            
                            if sb_gen_source == "ERCOT Asset":
                                src_type = "SCED" # Default
                                ast_name = sb_selected_asset
                                if selected_asset_info:
                                    res_id = selected_asset_info.get("resource_name")
                                    ast_cap = selected_asset_info.get("capacity_mw")
                                
                                if sb_selected_asset == "Azure Sky Wind":
                                    if sb_az_source == "Bill (Actuals)":
                                        src_type = "BILL"
                                        bill_path = "AzureSkyActuals.xlsx"
                                    elif sb_az_source == "Modeled":
                                        src_type = "MODEL"
                                    else:
                                        src_type = "SCED"

                            # No need to check duplicates since we cleared the list
                            new_scenario = {
                                "id": datetime.now().isoformat() + f"_{added_count}",
                                "name": name,
                                "year": year,
                                "hub": hub,
                                "tech": tech,
                                "duration": s_duration,
                                "month": month,
                                "capacity_mw": s_capacity,
                                "vppa_price": s_vppa_price,
                                "no_curtailment": s_no_curtailment,
                                "revenue_share_pct": s_revenue_share_pct,
                                "force_tmy": s_force_tmy,
                                "wind_weather_source": s_wind_weather_source if tech == "Wind" else "AUTO",
                                "hrrr_forecast_hour": int(s_hrrr_forecast_hour) if tech == "Wind" else 0,
                                "wind_model_engine": s_wind_model_engine if tech == "Wind" else "STANDARD",
                                "custom_lat": s_custom_lat if s_use_custom_location else None,
                                "custom_lon": s_custom_lon if s_use_custom_location else None,
                                "custom_profile_path": None,
                                "source_type": src_type,
                                "resource_id": res_id,
                                "asset_name": ast_name,
                                "asset_capacity_mw": ast_cap,
                                "bill_file_path": bill_path,
                                "negative_price_floor": st.session_state.get("val_price_floor", -3.0) if st.session_state.get("val_use_price_floor", False) else None,
                        }
                        st.session_state.scenarios.append(new_scenario)
                        added_count += 1
            
            if added_count > 0:
                st.success(f"Generated {added_count} scenarios!")
                st.rerun()
            else:
                st.warning("No scenarios created.")
    
    if reset_all_button:
        # Logic handled in callback
        st.rerun()

# Manage Scenarios
with tab_scenarios:
    if st.session_state.scenarios:
        st.sidebar.markdown("---")
        
    
        
        st.sidebar.subheader("Current Scenarios")
        for i, sc in enumerate(st.session_state.scenarios):
            col1, col2 = st.sidebar.columns([0.85, 0.15])
            with col1:
                st.text(f"{i+1}. {sc['name']}")
            with col2:
                if st.button("❌", key=f"remove_{i}", help="Remove this scenario"):
                    st.session_state.scenarios.pop(i)
                    st.rerun()
    
    # --- Main Content ---
    
        if st.session_state.scenarios:
            
            # --- Monte Carlo Simulation Section ---
            st.markdown("---")
            st.subheader("🎲 Monte Carlo Simulation (Optional)")
            st.caption("Generate probabilistic outcomes by random sampling from historical weather and price data")
            
            with st.expander("📊 **Monte Carlo Settings**", expanded=False):
                mc_col1, mc_col2 = st.columns([2, 1])
                
                with mc_col1:
                    enable_monte_carlo = st.checkbox(
                        "Enable Probabilistic Analysis",
                        value=False,
                        help="Run thousands of scenarios by randomly sampling weather years (2005-2024) and price years (2020-2026)"
                    )
                
                with mc_col2:
                    n_iterations = st.number_input(
                        "Number of Iterations",
                        min_value=100,
                        max_value=10000,
                        value=1000,
                        step=100,
                        help="More iterations = smoother distribution but slower",
                        disabled=not enable_monte_carlo
                    )
                
                if enable_monte_carlo:
                    st.info("💡 **How it works:** For each iteration, we randomly select a weather year (e.g., 2014) and a price year (e.g., 2022), then calculate the VPPA settlement. After 1,000+ runs, we show you the P10/P50/P90 outcomes.")
                    
                    if st.button("🎲 Run Monte Carlo Analysis", type="primary", disabled=len(st.session_state.scenarios) == 0):
                        monte_carlo_results = {}
                        
                        progress_mc = st.progress(0)
                        status_text = st.empty()
                        
                        # PRE-LOAD SHARED CACHES (CRITICAL: Load once for all scenarios to ensure consistency)
                        # This ensures identical scenarios get identical Monte Carlo results
                        status_text.text("Pre-loading price data (shared across all scenarios)...")
                        price_cache = {}
                        price_years = list(range(2020, 2026))  # 2020-2025 (exclude 2026 - incomplete YTD data)
                        price_load_errors = []
                        
                        for price_year in price_years:
                            try:
                                price_cache[price_year] = load_market_data(price_year)
                            except Exception as e:
                                price_load_errors.append(f"{price_year}: {str(e)}")
                                st.error(f"❌ Failed to load price data for {price_year}: {e}")
                        
                        if price_load_errors:
                            st.error(f"❌ Failed to load {len(price_load_errors)} price years. Monte Carlo CANNOT run without price data!")
                            st.code("\n".join(price_load_errors))
                        else:
                            st.success(f"✅ Pre-loaded all {len(price_cache)} price years (2020-2025)")
                        
                        # PRE-LOAD GENERATION PROFILES (scenario-specific, but cached across scenarios with same config)
                        # Key: (tech, lat, lon, capacity_mw, turbine_type) -> {year: profile}
                        gen_cache_by_config = {}
                        
                        for scenario_idx, scenario in enumerate(st.session_state.scenarios):
                            status_text.text(f"Running Monte Carlo for: {scenario['name']}...")
                            
                            # Prepare scenario config for Monte Carlo
                            mc_config = {
                                'hub': scenario['hub'],
                                'tech': scenario['tech'],
                                'capacity_mw': scenario['capacity_mw'],
                                'lat': scenario.get('custom_lat') if scenario.get('custom_lat') is not None else HUB_LOCATIONS[scenario['hub']][0],
                                'lon': scenario.get('custom_lon') if scenario.get('custom_lon') is not None else HUB_LOCATIONS[scenario['hub']][1],
                                'vppa_price': scenario['vppa_price'],
                                'revenue_share': scenario.get('revenue_share_pct', 100),
                                'curtail_neg': scenario.get('curtailment', False),
                                'turbine_type': scenario.get('turbine', 'GENERIC'),
                                'wind_model_engine': scenario.get('wind_model_engine', 'STANDARD'),
                                'year': scenario['year']  # For reference
                            }
                            
                            # Create cache key for this scenario's generation config
                            cache_key = (
                                mc_config['tech'],
                                mc_config['lat'],
                                mc_config['lon'],
                                mc_config['capacity_mw'],
                                mc_config['turbine_type'],
                                mc_config.get('wind_model_engine', 'STANDARD'),
                            )
                            
                            # Check if we already loaded profiles for this config
                            if cache_key not in gen_cache_by_config:
                                status_text.text(f"Pre-loading generation profiles for {scenario['name']}...")
                                gen_cache = {}
                                weather_years = list(range(2005, 2025))  # 2005-2024
                                gen_load_errors = []
                                
                                for idx, wx_year in enumerate(weather_years):
                                    try:
                                        status_text.text(f"Loading generation profile {idx+1}/{len(weather_years)}: {wx_year}...")
                                        profile = fetch_tmy.get_profile_for_year(
                                            year=wx_year,
                                            tech=mc_config['tech'],
                                            lat=mc_config['lat'],
                                            lon=mc_config['lon'],
                                            capacity_mw=mc_config['capacity_mw'],
                                            force_tmy=False,
                                            turbine_type=mc_config['turbine_type'],
                                            efficiency=0.86,
                                            hub_name=mc_config.get('hub'),
                                            apply_wind_calibration=(mc_config['tech'] == "Wind"),
                                            wind_model_engine=mc_config.get('wind_model_engine', 'STANDARD'),
                                        )
                                        
                                        # CRITICAL: Validate the profile is not empty
                                        if profile is None:
                                            error_msg = f"{wx_year}: Profile is None (fetch_tmy returned None)"
                                            gen_load_errors.append(error_msg)
                                            st.error(f"❌ {error_msg}")
                                        elif len(profile) == 0:
                                            error_msg = f"{wx_year}: Profile is empty (0 rows). Check if weather data cache is corrupt or API failed."
                                            gen_load_errors.append(error_msg)
                                            st.error(f"❌ {error_msg}")
                                        else:
                                            # Profile is valid, add to cache
                                            gen_cache[wx_year] = profile
                                            
                                    except Exception as e:
                                        gen_load_errors.append(f"{wx_year}: {str(e)}")
                                        st.warning(f"Could not load generation profile for {wx_year}: {e}")
                                
                                if gen_load_errors:
                                    st.warning(f"⚠️ Failed to load {len(gen_load_errors)} generation profiles. Monte Carlo will use on-demand fetching for those years (slower).")
                                    st.code("\n".join(gen_load_errors[:10]))  # Show first 10 errors
                                else:
                                    st.success(f"✅ Pre-loaded all {len(gen_cache)} generation profiles")
                                
                                gen_cache_by_config[cache_key] = gen_cache # Store for reuse
                            else:
                                gen_cache = gen_cache_by_config[cache_key] # Reuse existing cache
                                st.info(f"✅ Reusing pre-loaded generation profiles for {scenario['name']}")
                            
                            # Define progress callback
                            def update_progress(current, total):
                                progress_mc.progress((scenario_idx + current/total) / len(st.session_state.scenarios))
                                status_text.text(f"Running Monte Carlo iteration {current}/{total} for {scenario['name']}...")
                            
                            # Run Monte Carlo simulation with cached data
                            status_text.text(f"Running {n_iterations} Monte Carlo iterations...")
                            
                            # Capture debug output
                            import io
                            import sys
                            debug_output = io.StringIO()
                            old_stdout = sys.stdout
                            sys.stdout = debug_output
                            
                            try:
                                results_df, stats = monte_carlo.run_bootstrap_simulation(
                                    scenario_config=mc_config,
                                    n_iterations=n_iterations,
                                    price_data_cache=price_cache,
                                    generation_profile_cache=gen_cache,
                                    progress_callback=update_progress
                                )
                                
                                # Restore stdout and show debug output
                                sys.stdout = old_stdout
                                debug_text = debug_output.getvalue()
                                if debug_text:
                                    with st.expander("🔍 Debug Output", expanded=True):
                                        st.code(debug_text, language="text")
                                
                                monte_carlo_results[scenario['name']] = (results_df, stats)
                                
                            except Exception as e:
                                # Restore stdout
                                sys.stdout = old_stdout
                                debug_text = debug_output.getvalue()
                                if debug_text:
                                    st.warning("🔍 Debug output before error:")
                                    st.code(debug_text, language="text")
                                
                                st.error(f"Monte Carlo failed for {scenario['name']}: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                                continue
                        
                        progress_mc.progress(1.0)
                        status_text.text("Monte Carlo analysis complete!")
                        
                        # Store results in session state
                        st.session_state['monte_carlo_results'] = monte_carlo_results
                        
                        # Show success message
                        successful_count = sum(1 for _, (_, stats) in monte_carlo_results.items() if stats)
                        if successful_count > 0:
                            st.success(f"✅ Completed {n_iterations} iterations for {successful_count} scenarios")
                        else:
                            st.error("❌ All Monte Carlo simulations failed. Check that price data is available for years 2020-2026.")
            
            # Display Monte Carlo Results (if available)
            if 'monte_carlo_results' in st.session_state and st.session_state['monte_carlo_results']:
                st.markdown("---")
                st.subheader("📊 Monte Carlo Results")
                
                mc_results = st.session_state['monte_carlo_results']
                
                # Filter out failed simulations
                valid_results = {name: (df, stats) for name, (df, stats) in mc_results.items() if stats}
                
                if not valid_results:
                    st.warning("⚠️ No valid Monte Carlo results to display. All simulations may have failed due to missing data.")
                else:
                    # Comparison table
                    st.markdown("### Probabilistic Outcome Comparison")
                    comparison_df = monte_carlo.compare_scenarios_monte_carlo(valid_results)
                    st.dataframe(comparison_df.style.format({
                        'P10 ($)': '${:,.0f}',
                        'P50 ($)': '${:,.0f}',
                        'P90 ($)': '${:,.0f}',
                        'Mean ($)': '${:,.0f}',
                        'Std Dev ($)': '${:,.0f}',
                        'P90-P10 Range ($)': '${:,.0f}'
                    }), use_container_width=True)
                    
                    # Distribution plots for each scenario
                    st.markdown("### Distribution Plots")
                    
                    for scenario_name, (results_df, stats) in valid_results.items():
                        with st.expander(f"📈 {scenario_name}", expanded=True):
                            col_chart, col_stats = st.columns([2, 1])
                            
                            with col_chart:
                                # Histogram with percentile markers
                                fig = go.Figure()
                                
                                # Histogram
                                fig.add_trace(go.Histogram(
                                    x=results_df['annual_settlement_$'],
                                    nbinsx=50,
                                    name='Distribution',
                                    marker_color='lightblue',
                                    opacity=0.7
                                ))
                                
                                # Add P10, P50, P90 lines
                                for percentile, color, label in [
                                    (stats['P10'], 'red', 'P10 (Conservative)'),
                                    (stats['P50'], 'green', 'P50 (Median)'),
                                    (stats['P90'], 'blue', 'P90 (Optimistic)')
                                ]:
                                    fig.add_vline(
                                        x=percentile,
                                        line_dash="dash",
                                        line_color=color,
                                        annotation_text=f"{label}: ${percentile:,.0f}",
                                        annotation_position="top"
                                    )
                                
                                fig.update_layout(
                                    title=f"Distribution of Annual Settlement - {scenario_name}",
                                    xaxis_title="Annual Settlement ($)",
                                    yaxis_title="Frequency",
                                    showlegend=False,
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col_stats:
                                st.markdown("**Percentile Summary**")
                                percentile_table = monte_carlo.format_percentile_table(stats)
                                st.dataframe(percentile_table, use_container_width=True, hide_index=True)
                                
                                st.markdown("**Additional Stats**")
                                st.metric("Mean", f"${stats['Mean']:,.0f}")
                                st.metric("Std Dev", f"${stats['StdDev']:,.0f}")
                                st.metric("Range", f"${stats['Max'] - stats['Min']:,.0f}")
            
            st.markdown("---")
    
            # Calculate Results
            results = []
            progress_bar = st.progress(0)
    
            for i, scenario in enumerate(st.session_state.scenarios):
                # Fetch Data
                df_rtm = load_market_data(scenario['year'])
                if df_rtm.empty:
                    st.warning(f"Could not fetch data for {scenario['name']}")
                    continue
            
                # Calculate
                try:
                    df_res = calculate_scenario(scenario, df_rtm)
                except Exception as e:
                    st.error(f"Scenario '{scenario['name']}' failed: {e}")
                    continue
        
                # Aggregates
                total_rev = df_res['Settlement_Amount'].sum()
                total_gen = df_res['Gen_Energy_MWh'].sum()
                total_curt = df_res['Curtailed_MWh'].sum()
                avg_price = df_res['SPP'].mean()
                capture_price = (df_res['SPP'] * df_res['Gen_Energy_MWh']).sum() / total_gen if total_gen > 0 else 0
        
                # Calculate Aggregates for Charts (Memory Optimization)
                # 1. Daily for Cumulative Chart
                daily_agg = df_res.set_index('Time_Central')[['Settlement_Amount']].resample('D').sum().cumsum().reset_index()
                # Normalize Date for Seasonal Plot
                daily_agg['Normalized_Date'] = daily_agg['Time_Central'].apply(lambda x: x.replace(year=2024))
        
                # 2. Monthly for Bar Charts
                df_res['Month'] = df_res['Time_Central'].dt.strftime('%b')
                df_res['Month_Num'] = df_res['Time_Central'].dt.month
                # Group by Month and Year (to keep unique months if spanning years, though current use case is 1 year)
                # Actually, we normalize monthly charts too.
                monthly_agg = df_res.groupby(['Month', 'Month_Num'], as_index=False)[['Settlement_Amount', 'Gen_Energy_MWh']].sum()
                monthly_agg['Normalized_Month_Date'] = pd.to_datetime(monthly_agg['Month_Num'].astype(str) + "-01-2024", format="%m-%d-%Y")
                # Restore Month_Date for insight text (using actual year)
                monthly_agg['Month_Date'] = pd.to_datetime(monthly_agg['Month_Num'].astype(str) + f"-01-{scenario['year']}", format="%m-%d-%Y")
        
                results.append({
                    "Scenario": scenario['name'],
                    "Year": scenario['year'],
                    "Hub": scenario['hub'],
                    "Tech": scenario['tech'],
                    "Capacity (MW)": scenario['capacity_mw'],
                    "VPPA Price ($/MWh)": scenario['vppa_price'],
                    "duration": scenario['duration'], # Track duration type for plotting
                    "Net Settlement ($)": total_rev,
                    "Total Gen (MWh)": total_gen,
                    "Curtailed (MWh)": total_curt,
                    "Capture Price ($/MWh)": capture_price,
                    "Avg Hub Price ($/MWh)": avg_price,
                    # "data": df_res # DROPPED for Memory Savings
                    "daily_agg": daily_agg,
                    "monthly_agg": monthly_agg
                })
                progress_bar.progress((i + 1) / len(st.session_state.scenarios))
    
            progress_bar.empty()
    
            # ... (Visualizations Logic is generic, so no changes needed in the middle block) ...
    
            # ... Skip to Data Preview block adjustments manually below ...
    
            # --- Visualizations ---
    
            # Custom Color Palette based on SustainRound
            # Primary Blue: #0171BB
            COLOR_SEQUENCE = [
                "#0171BB", # SustainRound Blue
                "#FFC107", # Amber (Solar)
                "#4CAF50", # Green (Wind/Sustainability)
                "#9C27B0", # Purple
                "#FF5722", # Deep Orange
                "#607D8B", # Blue Grey
                "#E91E63", # Pink
                "#795548", # Brown
            ]
    
    
            # 1. Summary Metrics
            st.subheader("Summary Metrics")
            
            # Check if any scenario is using month-to-date analysis and show notification
            mtd_scenarios = [s for s in st.session_state.scenarios if 'date_range_note' in s]
            if mtd_scenarios:
                note_text = mtd_scenarios[0]['date_range_note']
                st.info(f"ℹ️ **Current Month Analysis**: {note_text} (limited to available data)")
    
            # Filter results for display
            display_cols = ["Scenario", "Net Settlement ($)", "Total Gen (MWh)", "Curtailed (MWh)", "Capture Price ($/MWh)", "Avg Hub Price ($/MWh)"]
            df_summary = pd.DataFrame(results)[display_cols]
    
            # Format columns
            st.dataframe(
                df_summary.style.format({
                    "Net Settlement ($)": "${:,.0f}",
                    "Total Gen (MWh)": "{:,.0f}",
                    "Curtailed (MWh)": "{:,.0f}",
                    "Capture Price ($/MWh)": "${:.2f}",
                    "Avg Hub Price ($/MWh)": "${:.2f}"
                })
            )
    
            # Prepare Data for Plotly
            # We need long-format dataframes for Plotly Express
    
            st.subheader("Cumulative Settlement ($)")
    
            # Insight for Cumulative (using existing data from results)
            # Re-calculate best/worst based on final totals
            final_settlements = {r['Scenario']: r['Net Settlement ($)'] for r in results}
            best_scen = max(final_settlements, key=final_settlements.get)
            best_val = final_settlements[best_scen]
            worst_scen = min(final_settlements, key=final_settlements.get)
            worst_val = final_settlements[worst_scen]
    
            if len(final_settlements) > 1:
                st.markdown(
                    f"**Insight:** The **{best_scen}** scenario leads with a total settlement of "
                    f"**${best_val:,.0f}**, while **{worst_scen}** trails at **${worst_val:,.0f}**."
                )
            else:
                st.markdown(
                    f"**Insight:** The **{best_scen}** scenario has a total settlement of **${best_val:,.0f}**."
                )
    
            # Initialize Plotly Graph Object for improved flexibility
            fig_cum = go.Figure()
    
            for i, res in enumerate(results):
                # Use pre-calculated daily aggregate
                daily = res['daily_agg']
                scenario_name = res['Scenario']
                duration_type = res['duration']
                color = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]
        
                if duration_type == "Specific Month":
                    # Plot as a "Pin" (Marker + Text) at the end of the month
                    if not daily.empty:
                        last_point = daily.iloc[-1]
                
                        fig_cum.add_trace(go.Scatter(
                            x=[last_point['Normalized_Date']],
                            y=[last_point['Settlement_Amount']],
                            mode='markers+text',
                            name=scenario_name,
                            marker=dict(color=color, size=12, symbol='circle'),
                            text=[f"${last_point['Settlement_Amount']:,.0f}"],
                            textposition="top center",
                            hovertemplate=f"<b>{scenario_name}</b><br>Month Total: ${{y:,.0f}}<extra></extra>"
                        ))
                else:
                    # Plot as a Line for Full Year
                    fig_cum.add_trace(go.Scatter(
                        x=daily['Normalized_Date'],
                        y=daily['Settlement_Amount'],
                        mode='lines',
                        name=scenario_name,
                        line=dict(color=color, width=3),
                        hovertemplate="<b>%{x|%b %d}</b><br>Cumulative: $%{y:,.0f}<extra></extra>"
                    ))
    
            fig_cum.update_layout(
                title="Cumulative Settlement Over Time (Seasonal Comparison)",
                legend_title="Scenario",
                hovermode="x unified"
            )
    
            fig_cum.update_yaxes(tickprefix="$", title="Settlement Amount ($)")
    
            # Format x-axis to show only Month (e.g., Jan, Feb)
            # Force range to full year (2024)
            fig_cum.update_xaxes(
                title="Month", 
                tickformat="%b",
                dtick="M1",
                range=["2024-01-01", "2024-12-31"]
            )
    
            st.plotly_chart(fig_cum, use_container_width=True)
    
            # Monthly Data
            monthly_data = []
            for res in results:
                m_agg = res['monthly_agg'].copy()
                m_agg['Scenario'] = res['Scenario']
                monthly_data.append(m_agg)
    
            if monthly_data:
                df_monthly = pd.concat(monthly_data, ignore_index=True)
        
        
                # Toggle for Monthly vs Annual view
                settle_view_mode = st.radio("View Mode", ["Monthly", "Annual"], horizontal=True, key="settle_view_mode")
        
                if settle_view_mode == "Annual":
                    st.subheader("Annual Net Settlement ($)")
            
                    # Annual view: Sum by scenario
                    df_annual_settle = df_monthly.groupby('Scenario').agg({
                        'Settlement_Amount': 'sum'
                    }).reset_index()
            
                    # Insight for Annual
                    best_scen = df_annual_settle.loc[df_annual_settle['Settlement_Amount'].idxmax(), 'Scenario']
                    best_val = df_annual_settle['Settlement_Amount'].max()
            
                    st.markdown(
                        f"**Insight:** **{best_scen}** led with a total settlement of **${best_val:,.0f}**."
                    )
            
                    fig_settle = px.bar(
                        df_annual_settle,
                        x='Scenario',
                        y='Settlement_Amount',
                        color='Scenario',
                        title="Annual Net Settlement Comparison",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        text='Settlement_Amount'
                    )
                    fig_settle.update_traces(texttemplate='$%{text:,.0f}', textposition='outside', cliponaxis=False)
                    fig_settle.update_yaxes(title="Total Settlement ($)")
                    fig_settle.update_xaxes(title="Scenario")
                    fig_settle.update_layout(
                        showlegend=True, 
                        legend_title_text="Scenario",
                        margin=dict(t=60, b=60, l=60, r=60)
                    )
            
                    st.plotly_chart(fig_settle, use_container_width=True)
            
                else:
                    # Chart 2: Monthly Net Settlement
                    st.subheader("Monthly Net Settlement ($)")
            
                    # Insight for Monthly Settlement
                    best_month_row = df_monthly.loc[df_monthly['Settlement_Amount'].idxmax()]
                    worst_month_row = df_monthly.loc[df_monthly['Settlement_Amount'].idxmin()]
            
                    best_amount = best_month_row['Settlement_Amount']
                    best_month = best_month_row['Month_Date'].strftime('%B %Y')
                    best_scenario = best_month_row['Scenario']
                    
                    worst_amount = worst_month_row['Settlement_Amount']
                    worst_month = worst_month_row['Month_Date'].strftime('%B %Y')
                    worst_scenario = worst_month_row['Scenario']
            
                    st.markdown(
                        f"**Insight:** The highest monthly return was **${best_amount:,.0f}** "
                        f"in **{best_month}** ({best_scenario}), "
                        f"whereas the lowest was **${worst_amount:,.0f}** "
                        f"in **{worst_month}** ({worst_scenario})."
                    )
        
                    fig_settle = px.bar(
                        df_monthly, 
                        x='Normalized_Month_Date', 
                        y='Settlement_Amount', 
                        color='Scenario', 
                        barmode='group',
                        title="Monthly Net Settlement (Seasonal Comparison)",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        hover_data={"Normalized_Month_Date": False, "Month_Date": "|%b %Y"}
                    )
                    fig_settle.update_yaxes(tickprefix="$", title="Settlement Amount ($)")
                    fig_settle.update_xaxes(
                        title="Month", 
                        tickformat="%b", 
                        dtick="M1" # Force monthly ticks
                    )
                    st.plotly_chart(fig_settle, use_container_width=True)
        
        
                # Chart 3: Monthly Generation
                st.subheader("Monthly Generation (MWh)")
        
                # Toggle for Monthly vs Annual view
                view_mode = st.radio("View Mode", ["Monthly", "Annual"], horizontal=True, key="gen_view_mode")
        
                if view_mode == "Annual":
                    # Annual view: Sum by scenario
                    df_annual = df_monthly.groupby('Scenario').agg({
                        'Gen_Energy_MWh': 'sum'
                    }).reset_index()
            
                    # Extract Year from scenario name (assumes format "YYYY ...")
                    df_annual['Year'] = df_annual['Scenario'].str.extract(r'(\d{4})')[0]
            
                    # Create formatted text labels based on value magnitude
                    def format_mwh(value):
                        if value >= 1_000_000:
                            return f"{value/1_000_000:.1f}M"
                        elif value >= 100_000:
                            return f"{value/1000:.0f}k"
                        elif value >= 10_000:
                            return f"{value/1000:.1f}k"
                        else:
                            return f"{value:,.0f}"
            
                    df_annual['Text_Label'] = df_annual['Gen_Energy_MWh'].apply(format_mwh)
            
                    # Insight for Annual
                    max_gen_scen = df_annual.loc[df_annual['Gen_Energy_MWh'].idxmax(), 'Scenario']
                    max_gen_val = df_annual['Gen_Energy_MWh'].max()
            
                    st.markdown(
                        f"**Insight:** **{max_gen_scen}** was the top producer, generating **{max_gen_val:,.0f} MWh** annually.\n"
                    )
            
                    # Annual bar chart - Year on X-axis
                    fig_gen = px.bar(
                        df_annual,
                        x='Year',
                        y='Gen_Energy_MWh',
                        color='Scenario',
                        title="Annual Energy Generation Comparison",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        text='Text_Label',  # Use formatted labels
                        barmode='group'
                    )
            
                    # Style the text
                    fig_gen.update_traces(
                        textposition='outside',
                        textfont=dict(size=12, family="Arial, sans-serif"),
                        marker_line_width=0,
                        cliponaxis=False
                    )
            
                    # Format Y-axis with thousands separator
                    fig_gen.update_yaxes(
                        title="Annual Generation (MWh)",
                        tickformat=",.0f",
                        gridcolor='rgba(128, 128, 128, 0.2)'
                    )
            
                    fig_gen.update_xaxes(
                        title="Year", 
                        type='category',
                        tickfont=dict(size=13)
                    )
            
                    # Improve overall layout
                    fig_gen.update_layout(
                        showlegend=True, 
                        legend_title_text="Scenario",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12),
                        height=550,
                        bargap=0.15,
                        bargroupgap=0.1,
                        margin=dict(t=80, b=60, l=60, r=60)
                    )
            
                else:
                    # Monthly view (original)
                    # Insight for Generation
                    total_gen_by_scen = df_monthly.groupby('Scenario')['Gen_Energy_MWh'].sum()
                    max_gen_scen = total_gen_by_scen.idxmax()
                    max_gen_val = total_gen_by_scen.max()
            
                    st.markdown(
                        f"**Insight:** **{max_gen_scen}** was the top producer, generating **{max_gen_val:,.0f} MWh**.\n"
                    )
    
                    fig_gen = px.bar(
                        df_monthly, 
                        x='Normalized_Month_Date', 
                        y='Gen_Energy_MWh', 
                        color='Scenario', 
                        barmode='group',
                        title="Monthly Energy Generation (Seasonal Comparison)",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        hover_data={"Normalized_Month_Date": False, "Month_Date": "|%b %Y"}
                    )
                    fig_gen.update_yaxes(title="Generation (MWh)")
                    fig_gen.update_xaxes(
                        title="Month", 
                        tickformat="%b", 
                        dtick="M1"
                    )
        
                st.plotly_chart(fig_gen, use_container_width=True)
    
            # Data Preview
            with st.expander("Downloads"):
                if results:
                    # Scenario Selection
                    scenario_names = [res['Scenario'] for res in results]
                    selected_scenario_name = st.selectbox("Select Scenario", scenario_names)
            
                    st.info("Generating detailed data on demand to save memory...")
            
                    # Find selected result metadata
                    # We need to re-find the original scenario config from session_state
                    # because 'results' only has aggregates now.
                    selected_scenario_config = next(s for s in st.session_state.scenarios if s['name'] == selected_scenario_name)
            
                    # Re-calculate on demand
                    year_val = selected_scenario_config['year']
                    df_rtm = load_market_data(year_val)
                    if not df_rtm.empty:
                        try:
                            df_display = calculate_scenario(selected_scenario_config, df_rtm)
                        except Exception as e:
                            st.error(f"Could not generate detailed data for {selected_scenario_name}: {e}")
                            df_display = None
                        if df_display is None:
                            st.warning("Detailed interval export is unavailable for this scenario.")
                            df_display = pd.DataFrame()
                
                        st.markdown(f"**Showing data for: {selected_scenario_name}**")
                
                        # 1. Scenario Configuration Table
                        st.subheader("1. Scenario Configuration")
                        config_data = {
                            "Parameter": ["Year", "Hub", "Technology", "Capacity (MW)", "VPPA Price ($/MWh)", "Duration"],
                            "Value": [
                                selected_scenario_config.get('year'),
                                selected_scenario_config.get('hub'),
                                selected_scenario_config.get('tech'),
                                f"{selected_scenario_config.get('capacity_mw', 0):.1f}",
                                f"${selected_scenario_config.get('vppa_price', 0):.2f}",
                                selected_scenario_config.get('duration')
                            ]
                        }
                        st.table(pd.DataFrame(config_data))
                
                        # 2. Monthly Performance Table
                        st.subheader("2. Monthly Performance Details")
                
                        # We need to find the monthly aggregate for this scenario from the 'results' list
                        # The 'results' list has the 'monthly_agg' dataframe inside it
                        selected_res = next((r for r in results if r['Scenario'] == selected_scenario_name), None)
                
                        if selected_res and 'monthly_agg' in selected_res:
                            monthly_df = selected_res['monthly_agg'].copy().sort_values('Month_Num')
                            # Format columns for display
                            display_monthly = monthly_df[['Month', 'Settlement_Amount', 'Gen_Energy_MWh']].copy()
                            display_monthly.columns = ['Month', 'Net Settlement ($)', 'Generation (MWh)']
                    
                            # Add formatting
                            st.dataframe(display_monthly.style.format({
                                'Net Settlement ($)': '${:,.0f}',
                                'Generation (MWh)': '{:,.0f}'
                            }))
                        else:
                            st.info("Monthly aggregation data not available.")
    
                        # 3. Detailed Interval Data
                        st.subheader("3. Detailed Interval Data (Top 1000 Rows)")
                        st.dataframe(df_display.head(1000)) # Limit display rows
                
                        # Download CSV
                        csv = df_display.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Detailed CSV",
                            data=csv,
                            file_name=f"{selected_scenario_name}.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error("Could not load data.")
                
                    st.markdown("---")
            
                    # Download Summary as Excel
                    excel_buffer = _build_settlement_excel(results, df_summary, st.session_state.scenarios)
                    st.download_button(
                        label="Download Summary Excel",
                        data=excel_buffer,
                        file_name="vppa_settlement_summary.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            
                    st.markdown("---")
            
                    # Download PDF Report
                    st.subheader("📄 PDF Report")
                    st.markdown("Generate a comprehensive PDF report with summary metrics and all visualizations.")
            
                    if st.button("Generate PDF Report"):
                        with st.spinner("Generating PDF report..."):
                            try:
                                # Store the current chart figures
                                # We need to ensure charts are in Annual view mode for the PDF
                        
                                # Cumulative chart (already created above as fig_cum)
                                # Settlement chart - create annual version
                                df_annual_settle = df_monthly.groupby('Scenario').agg({
                                    'Settlement_Amount': 'sum'
                                }).reset_index()
                        
                                fig_settle_pdf = px.bar(
                                    df_annual_settle,
                                    x='Scenario',
                                    y='Settlement_Amount',
                                    color='Scenario',
                                    title="Annual Net Settlement Comparison",
                                    color_discrete_sequence=COLOR_SEQUENCE,
                                    text='Settlement_Amount'
                                )
                                fig_settle_pdf.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                                fig_settle_pdf.update_yaxes(title="Total Settlement ($)")
                                fig_settle_pdf.update_xaxes(title="Scenario")
                                fig_settle_pdf.update_layout(showlegend=True, legend_title_text="Scenario")
                        
                                # Generation chart - create annual version
                                df_annual_gen = df_monthly.groupby('Scenario').agg({
                                    'Gen_Energy_MWh': 'sum'
                                }).reset_index()
                                df_annual_gen['Year'] = df_annual_gen['Scenario'].str.extract(r'(\d{4})')[0]
                        
                                def format_mwh(value):
                                    if value >= 1_000_000:
                                        return f"{value/1_000_000:.1f}M"
                                    elif value >= 100_000:
                                        return f"{value/1000:.0f}k"
                                    elif value >= 10_000:
                                        return f"{value/1000:.1f}k"
                                    else:
                                        return f"{value:,.0f}"
                        
                                df_annual_gen['Text_Label'] = df_annual_gen['Gen_Energy_MWh'].apply(format_mwh)
                        
                                fig_gen_pdf = px.bar(
                                    df_annual_gen,
                                    x='Year',
                                    y='Gen_Energy_MWh',
                                    color='Scenario',
                                    title="Annual Energy Generation Comparison",
                                    color_discrete_sequence=COLOR_SEQUENCE,
                                    text='Text_Label',
                                    barmode='group'
                                )
                                fig_gen_pdf.update_traces(
                                    textposition='outside',
                                    textfont=dict(size=12, family="Arial, sans-serif"),
                                    marker_line_width=0
                                )
                                fig_gen_pdf.update_yaxes(
                                    title="Annual Generation (MWh)",
                                    tickformat=",.0f",
                                    gridcolor='rgba(128, 128, 128, 0.2)'
                                )
                                fig_gen_pdf.update_xaxes(
                                    title="Year",
                                    type='category',
                                    tickfont=dict(size=13)
                                )
                                fig_gen_pdf.update_layout(
                                    showlegend=True,
                                    legend_title_text="Scenario",
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(size=12),
                                    height=550,
                                    bargap=0.15,
                                    bargroupgap=0.1,
                                    margin=dict(t=80, b=60, l=60, r=20)
                                )
                        
                                # Generate PDF
                                pdf_buffer = generate_pdf_report(results, df_summary)
                        
                                # Download button
                                report_date = datetime.now().strftime('%Y%m%d_%H%M%S')
                                st.download_button(
                                    label="📥 Download PDF Report",
                                    data=pdf_buffer.getvalue(),
                                    file_name=f"vppa_report_{report_date}.pdf",
                                    mime="application/pdf"
                                )
                        
                                st.success("✅ PDF report generated successfully!")
                        
                            except Exception as e:
                                st.error(f"Error generating PDF: {str(e)}")
                                st.info("Make sure all required dependencies are installed: `pip install reportlab kaleido Pillow`")
    
with tab_validation:
    st.header("Bill Validation")
    st.markdown("Create settlement bills based on weather models or validate incoming bills against official market prices. (Beta)")

    # --- Debug / Diagnostics ---
    with st.expander("🔧 Data Diagnostics (Troubleshooting)", expanded=True):
        c_dbg1, c_dbg2 = st.columns([0.5, 0.5])
        with c_dbg1:
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Reload Cache", key="btn_force_reload_val", help="Clear cache and reload file from disk"):
                    st.cache_data.clear()
                    st.rerun()
            with col_btn2:
                if st.button("Fetch Updates", key="btn_fetch_ercot", help="Download latest data from ERCOT"):
                    with st.spinner("Updating..."):
                        try:
                            repo_dir = str(Path(__file__).resolve().parent)
                            proc = subprocess.Popen(
                                [sys.executable, "update_ercot_2026.py"],
                                cwd=repo_dir,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                            )

                            log_lines = []
                            log_placeholder = st.empty()
                            start_ts = time.time()
                            timeout_sec = 300

                            while True:
                                if proc.stdout is not None:
                                    line = proc.stdout.readline()
                                    if line:
                                        log_lines.append(line.rstrip())
                                        log_placeholder.code("\n".join(log_lines[-25:]))

                                if proc.poll() is not None:
                                    break

                                if (time.time() - start_ts) > timeout_sec:
                                    proc.kill()
                                    raise TimeoutError(
                                        f"Update timed out after {timeout_sec} seconds. "
                                        "Try again, or run `python update_ercot_2026.py` in terminal."
                                    )

                                time.sleep(0.1)

                            if proc.returncode != 0:
                                tail = "\n".join(log_lines[-15:]) if log_lines else "No output captured."
                                raise RuntimeError(f"Updater failed (exit {proc.returncode}).\n{tail}")

                            st.cache_data.clear()
                            st.success("ERCOT 2026 file updated.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")
        with c_dbg2:
            try:
                fpath = "ercot_rtm_2026.parquet"
                if os.path.exists(fpath):
                    mtime = os.path.getmtime(fpath)
                    dt = datetime.fromtimestamp(mtime)
                    st.write(f"**File:** `{fpath}` | **Modified:** {dt.strftime('%H:%M:%S')}")
                    # Peek at file
                    meta_df = pd.read_parquet(fpath, columns=['Time_Central', 'date'])
                    date_range = f"{meta_df['date'].min()} to {meta_df['date'].max()}" if 'date' in meta_df.columns else f"{meta_df['Time_Central'].min().date()} to {meta_df['Time_Central'].max().date()}"
                    st.write(f"**Range:** {date_range} ({len(meta_df):,} rows)")
                    max_ts = pd.to_datetime(meta_df['Time_Central'], errors='coerce').max()
                    if pd.notna(max_ts):
                        lag_days = (pd.Timestamp.now(tz='US/Central') - max_ts).total_seconds() / 86400.0
                        st.write(f"**Lag vs now:** {lag_days:.1f} days")
                        if lag_days > 3:
                            st.caption("Note: ERCOT/public feed can be delayed; this may still be the latest available data.")
                else:
                    st.error("File not found!")
            except Exception as e:
                st.error(f"Error: {e}")

    # --- Configuration & Preview Header ---
    st.subheader("⚙️ Analysis Configuration")
    st.caption("Configure market parameters and project profiles for validation")

    # --- Integrated Configuration & Preview Controls ---
    with st.container():
        if 'val_custom_lat' not in st.session_state: st.session_state.val_custom_lat = 32.0
        if 'val_custom_lon' not in st.session_state: st.session_state.val_custom_lon = -100.0
        if 'val_hub' not in st.session_state: st.session_state.val_hub = "HB_NORTH"

        # Load Asset Registry
        @st.cache_data
        def load_asset_registry(path_str: str, file_mtime_ns: int):
            # Include file mtime in cache key so dropdown refreshes after asset updates.
            try:
                with open(path_str, 'r') as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

        @st.cache_data
        def load_vppa_offtake_lookup(workbook_specs):
            """
            Build {project_name: offtake_mw} from local VPPA_8760* workbooks.
            workbook_specs: tuple[(path_str, file_mtime_ns)]
            """
            lookup = {}
            for path_str, file_mtime_ns in workbook_specs:
                try:
                    _, profile_cols, _, offtake_map = tab_vppa_8760_compare.load_vppa_summary_profiles_from_path(
                        str(path_str),
                        2026,
                        int(file_mtime_ns),
                    )
                except Exception:
                    continue

                for name in profile_cols:
                    mw = offtake_map.get(name)
                    if mw is None:
                        continue
                    try:
                        mw_f = float(mw)
                    except (TypeError, ValueError):
                        continue
                    if name not in lookup:
                        lookup[name] = mw_f
            return lookup

        assets_path = REPO_ROOT / "ercot_assets.json"
        assets_mtime_ns = assets_path.stat().st_mtime_ns if assets_path.exists() else -1
        asset_registry = load_asset_registry(str(assets_path), int(assets_mtime_ns))
        local_vppa_workbooks = tab_vppa_8760_compare._list_local_workbooks()
        vppa_workbook_specs = tuple((str(p), int(p.stat().st_mtime_ns)) for p in local_vppa_workbooks)
        vppa_offtake_lookup = load_vppa_offtake_lookup(vppa_workbook_specs)
        # sort by name
        asset_names = sorted(list(asset_registry.keys()))
        asset_names.append("Settlement Invoice")

        def update_loc_from_hub():
            # Only update if custom location is NOT checked (acting as a lock)
            if not st.session_state.get('val_use_custom_location', False):
                h_lat, h_lon = HUB_LOCATIONS.get(st.session_state.get('val_hub', "HB_NORTH"), (32.0, -100.0))
                st.session_state.val_custom_lat = h_lat
                st.session_state.val_custom_lon = h_lon
                st.session_state.val_map_lat = h_lat
                st.session_state.val_map_lon = h_lon

        # Row 0: Project Source Selection
        col_src, col_empty = st.columns([1, 3])
        with col_src:
            val_source = st.radio("Project Source", ["Generic / Hub", "Specific Project"], horizontal=True, key="val_source")

        # Apply Specific Project defaults when user switches modes.
        prev_source = st.session_state.get("val_prev_project_source", val_source)
        if val_source == "Specific Project":
            default_project = "Azure Sky Wind"
            if prev_source != "Specific Project" or st.session_state.get("val_project_name") not in asset_names:
                if default_project in asset_registry:
                    st.session_state["val_project_name"] = default_project
                elif asset_names:
                    st.session_state["val_project_name"] = asset_names[0]
            if prev_source != "Specific Project":
                st.session_state["val_price"] = 17.34
                st.session_state["val_wind_model_engine_label"] = "Standard (Current)"
        elif prev_source != "Generic / Hub":
            st.session_state["val_wind_model_engine_label"] = "Standard (Current)"
        st.session_state["val_prev_project_source"] = val_source

        # Row 1: Hub/Project & Year & Price
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
        
        selected_project_meta = {}
        selected_resource_id = None
        selected_project_name = st.session_state.get("val_project_name", "")
        
        with c1:
            if val_source == "Generic / Hub":
                # Hub Selection with Callback
                val_hub = st.selectbox("Select Hub", list(HUB_LOCATIONS.keys()), key="val_hub", on_change=update_loc_from_hub)
                # Update map location if not using custom location
                if not st.session_state.get('val_use_custom_location', False):
                    lat, lon = HUB_LOCATIONS.get(val_hub, (32.0, -100.0))
                    # Update map session state if not custom
                    st.session_state.val_map_lat = lat
                    st.session_state.val_map_lon = lon
                else:
                    lat = st.session_state.get('val_custom_lat', 32.0)
                    lon = st.session_state.get('val_custom_lon', -100.0)
                    st.session_state.val_map_lat = lat
                    st.session_state.val_map_lon = lon
            else:
                # Specific Project Selection
                val_project_name = st.selectbox("Select Project", asset_names, key="val_project_name")
                selected_project_name = val_project_name
                if val_project_name in asset_registry:
                    selected_project_meta = asset_registry[val_project_name]
                    # Auto-update location for map
                    st.session_state.val_map_lat = selected_project_meta.get('lat', 32.0)
                    st.session_state.val_map_lon = selected_project_meta.get('lon', -100.0)
                    st.session_state.val_custom_lat = selected_project_meta.get('lat', 32.0)
                    st.session_state.val_custom_lon = selected_project_meta.get('lon', -100.0)

                    selected_resource_id = selected_project_meta.get("resource_name")
                    first_sced_date = get_first_cached_sced_date(selected_resource_id)
                    if first_sced_date:
                        st.caption(f"SCED first available: {first_sced_date}")
                    elif selected_resource_id:
                        st.caption("SCED first available: not cached locally")
                    
                    # Auto-select Hub if possible (though we might want to keep it editable?)
                    # Ideally, we key off the project's node or just its lat/lon for price? 
                    # For now, let's keep Hub selectable or auto-select closest Hub.
                    # Project meta has 'hub' field usually.
                    proj_hub = selected_project_meta.get('hub')
                    # Map friendly name to ID if needed
                    hub_map_rev = {"North": "HB_NORTH", "South": "HB_SOUTH", "West": "HB_WEST", "Houston": "HB_HOUSTON", "Pan": "HB_PAN"}
                    if proj_hub in hub_map_rev:
                        # We can't easily programmatically set the selectbox value without rerunning...
                        # Just display it for now.
                        pass

        # Project Specific Defaults
        prev_project = st.session_state.get("val_prev_specific_project_name")
        if val_source == "Specific Project":
            is_new_selection = (prev_source != "Specific Project" or prev_project != selected_project_name)
            if is_new_selection:
                if selected_project_name == "Azure Sky Wind":
                    st.session_state["val_year"] = 2025
                    st.session_state["val_hub"] = "HB_NORTH" # Azure Sky is North
                elif selected_project_name == "Stafford Solar":
                    st.session_state["val_price"] = 42.55
                    st.session_state["val_hub"] = "HB_WEST" # Stafford settles West
            st.session_state["val_prev_specific_project_name"] = selected_project_name

        with c2:
            val_year = st.selectbox("Year", [2026, 2025, 2024, 2023, 2022, 2021, 2020], key="val_year")
        with c3:
            val_vppa_price = st.number_input("VPPA / Strike Price ($/MWh)", value=50.0, step=0.5, key="val_price")
        with c4:
            val_revenue_share = st.number_input(
                "Buyer's Upside Share %",
                min_value=0, max_value=100, value=100, step=5,
                help="% of upside buyer receives when SPP > PPA price.",
                key="val_revenue_share"
            )
            
        with st.expander("Advanced Settlement Options", expanded=False):
            c_adv0, c_adv1, c_adv2 = st.columns(3)
            with c_adv0:
                curtail_neg = st.checkbox("Curtail when Price < $0", value=True, help="Set Generation to 0 MWh when Hub Price is negative", key="val_curtail_neg")
            with c_adv1:
                val_use_price_floor = st.checkbox("Apply Negative Price Floor", value=False, help="Limits the downside of negative SPP prices if the project curtails or has a floor hedge.", key="val_use_price_floor")
            with c_adv2:
                val_price_floor = st.number_input("Floor SPP Price ($/MWh)", value=-3.0, step=1.0, disabled=not val_use_price_floor, key="val_price_floor")

        # Month selector
        _ALL_MONTHS = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        selected_month_names = st.multiselect(
            "Months",
            _ALL_MONTHS,
            default=_ALL_MONTHS,
            key="val_month_multiselect",
            help="Select which months to include in the analysis.",
        )
        _month_name_to_num = {m: i + 1 for i, m in enumerate(_ALL_MONTHS)}
        selected_month_numbers = sorted(_month_name_to_num[m] for m in selected_month_names)
        if not selected_month_numbers:
            st.warning("⚠️ No months selected. Please select at least one month.")

        # Row 2: Technology & Preview Settings
        c5, c6, c7 = st.columns(3)
        
        # Determine defaults based on selection
        default_tech = "Solar"
        default_cap = 100.0
        
        if val_source == "Specific Project" and selected_project_meta:
            default_tech = selected_project_meta.get('tech', 'Solar')
            default_cap = selected_project_meta.get('capacity_mw', 100.0)
        elif val_source == "Specific Project" and selected_project_name == "Settlement Invoice":
            default_tech = "Wind"
            default_cap = 350.0

        with c5:
            if val_source == "Generic / Hub":
                preview_tech = st.selectbox("Technology", ["Solar", "Wind"], key="preview_tech")
            else:
                st.text_input("Technology", value=default_tech, disabled=True, key=f"preview_tech_display_{selected_project_name}")
                preview_tech = default_tech # Use project tech
                preview_tech = default_tech # Use project tech

        with c6:
            # Capacity / Settlement MW
            if val_source == "Generic / Hub":
                preview_capacity = st.number_input("Capacity (MW)", min_value=1.0, max_value=1000.0, value=100.0, step=10.0, key="preview_capacity")
            else:
                # Settlement MW Logic
                project_total_cap = selected_project_meta.get('capacity_mw', 100.0)
                selected_offtake_mw = vppa_offtake_lookup.get(selected_project_name)
                prev_project_for_mw = st.session_state.get("val_prev_project_for_mw")
                if selected_project_name != prev_project_for_mw:
                    if selected_offtake_mw is not None:
                        st.session_state["val_settlement_mw"] = float(selected_offtake_mw)
                    elif "val_settlement_mw" not in st.session_state:
                        st.session_state["val_settlement_mw"] = float(min(68.0, project_total_cap))
                    st.session_state["val_prev_project_for_mw"] = selected_project_name

                default_settlement_mw = float(
                    st.session_state.get("val_settlement_mw", min(68.0, project_total_cap))
                )
                default_settlement_mw = max(0.1, default_settlement_mw)
                max_settlement_mw = float(max(project_total_cap * 1.5, default_settlement_mw))
                preview_capacity = st.number_input(
                    "Settlement Capacity (MW)", 
                    min_value=0.1, 
                    max_value=max_settlement_mw, # Allow some buffer, but warn?
                    value=default_settlement_mw,
                    step=1.0, 
                    help=f"Enter the MW volume to settle. Project Total: {project_total_cap} MW",
                    key="val_settlement_mw"
                )
                if selected_offtake_mw is not None:
                    st.caption(f"VPPA Offtake default: {selected_offtake_mw:,.2f} MW")
                if preview_capacity != project_total_cap:
                    pct = (preview_capacity / project_total_cap) * 100.0
                    st.caption(f"Settling {pct:.1f}% of Project ({project_total_cap} MW)")

        with c7:
            # Historical Weather Logic
            # Historical Weather Logic
            weather_options = ["Expected Production Based on Actual Weather", "Model Based on Weather", "Typical Meteorological Year"]
            
            # --- CUSTOM: Add Settlement Invoice if Azure Sky ---
            if val_source == "Specific Project" and "Azure Sky" in selected_project_name:
                 weather_options.insert(1, "Settlement Invoice (Actuals)")
                 weather_options.insert(3, "Actual SCED + Settlement Invoice")
                 weather_options.insert(4, "Actual SCED + Model Based on Weather + Settlement Invoice")
            
            preview_weather = st.selectbox("Data Source", weather_options, key="preview_weather")
            with st.expander("Key"):
                st.markdown("""
                - **Expected Production Based on Actual Weather**: The estimated generation derived from observed meteorological conditions.
                - **Model Based on Weather**: The generated model for production based on actual weather conditions, compared with actual SCED data.
                - **Typical Meteorological Year**: A typical long-term profile of weather conditions used for baseline comparisons.
                - **SCED**: Security Constrained Economic Dispatch, the actual realtime instructions ERCOT gives to generators.
                """)

        preview_wind_weather_source = "AUTO"
        preview_hrrr_forecast_hour = 0
        preview_wind_model_engine = "STANDARD"
        preview_apply_bp_cap = bool(st.session_state.get("val_apply_bp_cap", True))
        # Optional Turbine Selector (if Wind)
        selected_turbine = "GENERIC"
        # Only show turbine selector for Generic
        if val_source == "Generic / Hub" and preview_tech == "Wind":
            turbine_opts = ["Generic (IEC Class 2)", "Vestas V163 (Low Wind)", "GE 2.x (Workhorse)", "GE 3.6-154 (Modern Mainstream)", "Nordex N163 (5.X MW)", "Siemens-Gamesa SG 3.4-132"]
            c_turb1, c_turb2, c_turb3 = st.columns(3)
            with c_turb1:
                val_turb_ui = st.selectbox("Turbine Model", turbine_opts, key="val_preview_turbine")
            
            turbine_override_map = {
                "Generic (IEC Class 2)": "GENERIC",
                "Vestas V163 (Low Wind)": "VESTAS_V163",
                "GE 2.x (Workhorse)": "GE_2X",
                "GE 3.6-154 (Modern Mainstream)": "GE_3X",
                "Nordex N163 (5.X MW)": "NORDEX_N163",
                "Siemens-Gamesa SG 3.4-132": "SG_3_4_132",
            }
            if val_turb_ui in turbine_override_map:
                selected_turbine = turbine_override_map[val_turb_ui]
        
        elif val_source == "Specific Project" and preview_tech == "Wind":
            # Just show what we're using
            t_model = selected_project_meta.get('turbine_model', 'Generic')
            if 'turbines' in selected_project_meta:
                t_model = "Mixed Fleet (Blended Profile)"
            
            # We don't allow changing it for specific projects
            # But we need to pass the correct turbine type to the backend if simple
            # If complex (turbines list), fetch_tmy handles it.
            # If simple, we need to map the model string to our internal types ENUM if possible, 
            # OR fetch_tmy needs to be smart enough. 
            # For now, let's just display it.
            st.caption(f"Turbine: {t_model}")

        if preview_tech == "Wind":
            w1, w2, w3 = st.columns([2, 1, 2])
            with w1:
                preview_wind_label = st.selectbox(
                    "Wind Weather Dataset",
                    list(WIND_WEATHER_SOURCE_OPTIONS.keys()),
                    key="val_wind_weather_source_label",
                    help="NOAA HRRR uses local cached files under data_cache/hrrr.",
                )
                preview_wind_weather_source = WIND_WEATHER_SOURCE_OPTIONS.get(preview_wind_label, "AUTO")
            with w2:
                if preview_wind_weather_source == "NOAA_HRRR_CACHED":
                    preview_hrrr_forecast_hour = int(
                        st.number_input(
                            "HRRR fxx",
                            min_value=0,
                            max_value=18,
                            value=0,
                            step=1,
                            key="val_hrrr_forecast_hour",
                        )
                    )
                    st.caption("0=analysis")
                    hrrr_count = get_hrrr_cache_count()
                    if hrrr_count == 0:
                        st.warning("No HRRR cache files found yet. Generate them with `scripts/fetch_hrrr_wind.py`.")
                    else:
                        st.caption(f"Detected {hrrr_count} cached HRRR files.")
            with w3:
                preview_engine_label = st.selectbox(
                    "Wind Model Engine",
                    list(WIND_MODEL_ENGINE_OPTIONS.keys()),
                    index=0,
                    key="val_wind_model_engine_label",
                    help="Advanced mode applies monthly EIA/CF targets, SCED bias correction, node adjustments, and tuned clipping.",
                )
                preview_wind_model_engine = WIND_MODEL_ENGINE_OPTIONS.get(preview_engine_label, "STANDARD")
            if val_source == "Specific Project":
                preview_apply_bp_cap = st.toggle(
                    "Apply SCED Base Point Cap",
                    value=bool(st.session_state.get("val_apply_bp_cap", True)),
                    key="val_apply_bp_cap",
                    help=(
                        "Toggle modeled wind between capped and uncapped output when "
                        "SCED Base Point data is available."
                    ),
                )

        # On-demand SCED cache refresh for Azure Sky Wind so preview can stay cache-only and fast.
        if (
            val_source == "Specific Project"
            and selected_project_name == "Azure Sky Wind"
            and preview_weather in [
                "Actual SCED + Model",
                "Actual SCED + Settlement Invoice",
                "Actual SCED + Model + Settlement Invoice",
            ]
        ):
            azure_resource_id = selected_project_meta.get("resource_name")
            b1, b2 = st.columns([1, 2.5])
            with b1:
                if st.button(
                    "⬇️ Cache SCED Data (Azure)",
                    key=f"cache_azure_sced_{val_year}",
                    use_container_width=True,
                    help="Fetch/refresh Azure Sky SCED cache for the selected year. This can take several minutes.",
                ):
                    if not azure_resource_id:
                        st.error("Missing resource ID for Azure Sky Wind.")
                    else:
                        with st.spinner(f"Caching SCED data for {selected_project_name} ({val_year})..."):
                            df_cached = get_cached_asset_data_with_base_point(
                                azure_resource_id,
                                f"{val_year}-01-01",
                                f"{val_year}-12-31",
                            )
                            try:
                                sced_fetcher.consolidate_year(azure_resource_id, val_year)
                            except Exception:
                                pass
                            # Preview path uses local cache-only loader; clear cached results so new files are picked up.
                            get_cached_asset_data_with_base_point_local.clear()

                        if df_cached.empty:
                            st.warning("No SCED rows were cached for this year.")
                        else:
                            bp_rows = int(df_cached["Base_Point_MW"].notna().sum()) if "Base_Point_MW" in df_cached.columns else 0
                            st.success(
                                f"Cached {len(df_cached):,} SCED intervals for {val_year}. "
                                f"Base Point available on {bp_rows:,} intervals."
                            )
            with b2:
                st.caption(
                    "Use this once per year before `Generate Preview` when running SCED-based weather modes."
                )

        # On-demand Bill Data refresh
        is_invoice_selected = (
            (val_source == "Specific Project" and selected_project_name == "Settlement Invoice") or
            ("Settlement Invoice" in preview_weather)
        )
        
        if is_invoice_selected:
             b1, b2 = st.columns([1, 2.5])
             with b1:
                 if st.button("🔄 Refresh Invoice Data", key="refresh_bill_data", use_container_width=True):
                     with st.spinner("Updating Settlement Invoice data from Excel..."):
                        try:
                            import subprocess
                            subprocess.run(
                                [sys.executable, "scripts/convert_bill_to_parquet.py"],
                                check=True,
                                cwd=str(REPO_ROOT),
                            )
                            st.success("Settlement Invoice data updated!")
                        except Exception as e:
                            st.error(f"Failed to update bill data: {e}")
             with b2:
                 st.caption(
                     "Reloads data from `AzureSkyActuals.xlsx` and applies a 1-interval "
                     "price alignment correction for invoice settlement price."
                 )


        # Row 3: Actions
        c8, c9 = st.columns([3, 1])
        with c8:
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        with c9:
             if st.button("📈 Generate Preview", type="primary", use_container_width=True):
                with st.spinner("Generating profile..."):
                    try:
                        # Determine HUB for Price Logic
                        # If Generic, use val_hub. If Specific, use Project's Hub (map to ID).
                        calc_hub = st.session_state.get('val_hub', 'HB_NORTH')  # Default from session state
                        if val_source == "Specific Project" and selected_project_meta:
                            proj_hub_name = selected_project_meta.get('hub', 'North')
                            hub_map_rev = {"North": "HB_NORTH", "South": "HB_SOUTH", "West": "HB_WEST", "Houston": "HB_HOUSTON", "Pan": "HB_PAN"}
                            calc_hub = hub_map_rev.get(proj_hub_name, "HB_NORTH")
                        
                        # Market Data (hub-filtered cache to reduce validation latency)
                        df_market_hub = load_market_hub_data(val_year, calc_hub)
                        if df_market_hub.empty:
                            st.error(f"No market data for {val_year}")
                        else:
                            # Location handling
                            if val_source == "Specific Project":
                                lat = selected_project_meta.get('lat', 32.0)
                                lon = selected_project_meta.get('lon', -100.0)
                            else:
                                # Generic/Custom
                                lat, lon = st.session_state.val_custom_lat, st.session_state.val_custom_lon

                            weather_opts = []
                            median_year = None
                            project_turbine_model = "GENERIC"
                            if (
                                val_source == "Specific Project"
                                and preview_tech == "Wind"
                                and selected_project_meta
                                and ("turbines" not in selected_project_meta)
                            ):
                                project_turbine_model = str(
                                    selected_project_meta.get("turbine_model", selected_turbine or "GENERIC")
                                )
                            
                            # Pre-calculate P50 if needed
                            if preview_weather == "Expected Production Based on Actual Weather": 
                                weather_opts = [{"name": "Actual", "force_tmy": False, "year_override": None}]
                                if val_source == "Specific Project" and selected_project_name == "Settlement Invoice":
                                     weather_opts = [{"name": "Invoice", "source": "BILL"}]
                            elif preview_weather == "Settlement Invoice (Actuals)":
                                weather_opts = [{"name": "Invoice", "source": "BILL"}]
                            elif preview_weather == "Model Based on Weather":
                                # NEW: Fetch actual SCED data + generate model for comparison
                                weather_opts = [
                                    {"name": "SCED_Actual", "force_tmy": False, "year_override": None, "use_sced": True},
                                    {"name": "Model", "force_tmy": False, "year_override": None, "use_sced": False}
                                ]
                            elif preview_weather == "Actual SCED + Settlement Invoice":
                                weather_opts = [
                                    {"name": "SCED_Actual", "force_tmy": False, "year_override": None, "use_sced": True},
                                    {"name": "Invoice", "source": "BILL"},
                                ]
                            elif preview_weather == "Actual SCED + Model Based on Weather + Settlement Invoice":
                                weather_opts = [
                                    {"name": "SCED_Actual", "force_tmy": False, "year_override": None, "use_sced": True},
                                    {"name": "Model", "force_tmy": False, "year_override": None, "use_sced": False},
                                    {"name": "Invoice", "source": "BILL"},
                                ]
                            elif preview_weather == "Typical Meteorological Year": 
                                weather_opts = [{"name": "TMY", "force_tmy": True, "year_override": None}]

                            sced_basepoint = pd.DataFrame(columns=["Time_Central", "Base_Point_MW"])
                            bp_headroom_factor = 1.0
                            if (
                                val_source == "Specific Project"
                                and selected_project_meta
                                and preview_tech == "Wind"
                            ):
                                resource_id_ctx = selected_project_meta.get("resource_name")
                                if resource_id_ctx:
                                    try:
                                        df_sced_ctx = get_cached_asset_data_with_base_point_local(
                                            resource_id_ctx,
                                            f"{val_year}-01-01",
                                            f"{val_year}-12-31",
                                        )
                                        if (
                                            (not df_sced_ctx.empty)
                                            and ("Time" in df_sced_ctx.columns)
                                            and ("Base_Point_MW" in df_sced_ctx.columns)
                                        ):
                                            project_total = float(selected_project_meta.get("capacity_mw", preview_capacity) or preview_capacity)
                                            bp_headroom_factor = 1.0
                                            bp_scale = (preview_capacity / project_total) if project_total > 0 else 1.0
                                            sced_basepoint = df_sced_ctx[["Time", "Base_Point_MW"]].copy()
                                            sced_basepoint["Time"] = pd.to_datetime(
                                                sced_basepoint["Time"],
                                                utc=True,
                                                errors="coerce",
                                            )
                                            sced_basepoint["Base_Point_MW"] = (
                                                pd.to_numeric(sced_basepoint["Base_Point_MW"], errors="coerce") * float(bp_scale)
                                            )
                                            sced_basepoint = sced_basepoint.dropna(subset=["Time"])
                                            sced_basepoint["Time_Central"] = sced_basepoint["Time"].dt.tz_convert("US/Central")
                                            sced_basepoint = sced_basepoint[
                                                ["Time_Central", "Base_Point_MW"]
                                            ].drop_duplicates(subset=["Time_Central"])
                                            sced_basepoint["Base_Point_Effective_MW"] = (
                                                pd.to_numeric(sced_basepoint["Base_Point_MW"], errors="coerce")
                                                * float(bp_headroom_factor)
                                            )
                                            sced_basepoint["Base_Point_Effective_MW"] = sced_basepoint["Base_Point_Effective_MW"].clip(
                                                lower=0.0,
                                                upper=float(preview_capacity),
                                            )
                                    except Exception:
                                        # Keep valuation flow resilient even if SCED base point load fails.
                                        sced_basepoint = pd.DataFrame(columns=["Time_Central", "Base_Point_MW"])
                                        bp_headroom_factor = 1.0

                            preview_results = {}
                            for source in weather_opts:
                                target_year = source.get("year_override")
                                if target_year is None:
                                    target_year = val_year
                                
                                # CHECK IF WE SHOULD USE ACTUAL SCED DATA
                                use_sced = source.get("use_sced", False)
                                is_bill = source.get("source") == "BILL"
                                invoice_price_series = None

                                if is_bill:
                                    bill_xlsx = SETTLEMENT_INVOICE_XLSX
                                    bill_parquet_primary = SETTLEMENT_INVOICE_PARQUET
                                    bill_parquet_legacy = SETTLEMENT_INVOICE_PARQUET_LEGACY
                                    bill_parquet = (
                                        bill_parquet_primary
                                        if bill_parquet_primary.exists()
                                        else bill_parquet_legacy
                                    )
                                    
                                    if bill_xlsx.exists():
                                        stale = False
                                        if bill_parquet.exists():
                                            xlsx_mtime = bill_xlsx.stat().st_mtime
                                            parquet_mtime = bill_parquet.stat().st_mtime
                                            stale = xlsx_mtime > parquet_mtime
                                        else:
                                            stale = True
                                        
                                        if stale:
                                            with st.spinner("Updating Settlement Invoice data from Excel..."):
                                                try:
                                                    import subprocess
                                                    subprocess.run(
                                                        [sys.executable, "scripts/convert_bill_to_parquet.py"],
                                                        check=True,
                                                        cwd=str(REPO_ROOT),
                                                    )
                                                    st.toast("Settlement Invoice data updated!")
                                                except Exception as e:
                                                    st.error(f"Failed to update bill data: {e}")
                                            bill_parquet = (
                                                bill_parquet_primary
                                                if bill_parquet_primary.exists()
                                                else bill_parquet_legacy
                                            )
                                    
                                    # Load Settlement Invoice Parquet
                                    if not bill_parquet.exists():
                                        st.error(
                                            f"Settlement Invoice data not found at {bill_parquet_primary} "
                                            f"(legacy fallback: {bill_parquet_legacy}). Please run conversion script."
                                        )
                                        continue
                                    
                                    try:
                                        df_bill = pd.read_parquet(bill_parquet)
                                        # Filter by year
                                        df_bill = df_bill[df_bill['Time'].dt.year == val_year].copy()
                                        
                                        if df_bill.empty:
                                            st.warning(f"No Settlement Invoice data found for {val_year}.")
                                            continue
                                            
                                        # Set index and profile
                                        df_bill = df_bill.set_index('Time')
                                        if 'Settlement_Point_Price' in df_bill.columns:
                                            invoice_price_series = pd.to_numeric(
                                                df_bill['Settlement_Point_Price'],
                                                errors='coerce',
                                            )
                                            invoice_price_series = invoice_price_series.sort_index()
                                            if INVOICE_PRICE_ALIGNMENT_SHIFT_INTERVALS != 0:
                                                invoice_price_series = invoice_price_series.shift(
                                                    INVOICE_PRICE_ALIGNMENT_SHIFT_INTERVALS
                                                )
                                        
                                        # Resample/Fill if needed? Assuming 15-min.
                                        # Just take Actual_MW
                                        profile = df_bill['Actual_MW']
                                        profile.name = 'Gen_MW'
                                        
                                        # Scale?
                                        # If user entered Settlement MW, we might want to scale the bill data?
                                        # Usually bill data IS the settlement MW.
                                        # But if the bill is for 350MW and user wants to see 100MW share?
                                        # The UI has "Settlement Capacity".
                                        # Let's assume bill data is 100% of project.
                                        # Scale = (User_Settlement_MW / 350.0)?
                                        # Use the default cap we set (350.0).
                                        project_total = 350.0
                                        # Keep bill profile unscaled here; shared post-processing applies scale_factor once.
                                        scale_factor = preview_capacity / project_total if project_total > 0 else 1.0
                                        
                                    except Exception as e:
                                        st.error(f"Error loading bill data: {e}")
                                        continue

                                elif use_sced and val_source == "Specific Project":
                                    # Load actual ERCOT SCED data from cached parquet file
                                    resource_id = selected_project_meta.get('resource_name')
                                    if not resource_id:
                                        st.warning(
                                            f"SCED actuals unavailable: no `resource_name` found for "
                                            f"{selected_project_name or 'selected project'}."
                                        )
                                        continue

                                    try:
                                        df_sced = get_cached_asset_data_with_base_point_local(
                                            resource_id,
                                            f"{val_year}-01-01",
                                            f"{val_year}-12-31",
                                        )
                                    except Exception as e:
                                        st.error(f"Error loading SCED data: {e}")
                                        continue

                                    if df_sced.empty:
                                        st.warning(f"No cached SCED data found for {resource_id} in {val_year}.")
                                        st.info("SCED data must be pre-downloaded. Continuing with model-only output.")
                                        continue
                                    
                                    # Prepare SCED data for scaling
                                    # The common scaling block at L2470 will apply the scale_factor
                                    project_total = selected_project_meta.get('capacity_mw', 100.0)
                                    scale_factor = preview_capacity / project_total if project_total > 0 else 1.0
                                    
                                    # Create profile at full capacity scale
                                    df_sced = df_sced.set_index('Time')
                                    profile = df_sced['Actual_MW']
                                    profile.name = 'Gen_MW'
                                    
                                else:
                                    # SCALING LOGIC FOR SPECIFIC PROJECTS
                                    # For SCED comparison mode, use simplified model (skip complex mixed-fleet)
                                    # to avoid slow generation times
                                    fetch_capacity = preview_capacity
                                    scale_factor = 1.0
                                    turbines_config = None
                                    
                                    # Check if this is SCED comparison mode
                                    is_sced_comparison = preview_weather == "Model Based on Weather"
                                    
                                    if val_source == "Specific Project" and 'turbines' in selected_project_meta:
                                        # Use full mixed-fleet model (now enabled even for SCED comparison for higher accuracy)
                                        turbines_config = selected_project_meta['turbines']
                                        project_total = selected_project_meta.get('capacity_mw', 100.0)
                                        fetch_capacity = project_total # Fetch full plant
                                        if project_total > 0:
                                            scale_factor = preview_capacity / project_total
                                    
                                    # Handle turbine type for SCED comparison
                                    final_turbine = project_turbine_model if preview_tech == "Wind" else selected_turbine
                                    if is_sced_comparison and val_source == "Specific Project":
                                        # For Azure Sky specifically, or if turbine_model is Nordex
                                        t_model_raw = selected_project_meta.get('turbine_model', '').upper()
                                        selected_project_name_upper = str(
                                            selected_project_name or selected_project_meta.get("project_name", "")
                                        ).upper()
                                        if "NORDEX" in t_model_raw and "149" in t_model_raw:
                                            final_turbine = "NORDEX_N149"
                                        elif "AZURE" in selected_project_name_upper:
                                            final_turbine = "NORDEX_N149"
                                    
                                    # Show progress for potentially slow operations
                                    if turbines_config:
                                        model_desc = "mixed fleet (blended)"
                                    else:
                                        model_desc = f"Nordex N149" if (is_sced_comparison and final_turbine == "NORDEX_N149") else ("simplified generic" if is_sced_comparison else source['name'])
                                    
                                    with st.spinner(f"Generating {model_desc} profile ({fetch_capacity:.0f} MW)..."):
                                        profile = fetch_tmy.get_profile_for_year(
                                            year=target_year, 
                                            tech=preview_tech, 
                                            lat=lat, 
                                            lon=lon, 
                                            capacity_mw=fetch_capacity, 
                                            force_tmy=source["force_tmy"], 
                                            turbine_type=final_turbine,
                                            efficiency=0.86, 
                                            hub_name=calc_hub,
                                            apply_wind_calibration=(preview_tech == "Wind"),
                                            turbines=turbines_config,
                                            wind_weather_source=preview_wind_weather_source,
                                            hrrr_forecast_hour=preview_hrrr_forecast_hour,
                                            wind_model_engine=preview_wind_model_engine,
                                        )
                                    
                                if profile is not None:
                                    # Apply Scale Factor if needed
                                    if scale_factor != 1.0:
                                        profile = profile * scale_factor
                                    
                                    pc = profile.tz_convert('US/Central')
                                    
                                    # ROBUST ALIGNMENT LOGIC
                                    # Map the historical profile (e.g., 2009) to the target year (e.g., 2026)
                                    # This handles gaps in market data correctly, unlike array slicing logic
                                    
                                    # 1. Create source dataframe
                                    pdf = pd.DataFrame({'Gen_MW': pc.values, 'Time_Source': pc.index})
                                    
                                    # 2. Shift timestamps to target year (val_year)
                                    # We use the same day/month/time but change the year
                                    # Handle leap years: 2009 (non-leap) -> 2026 (non-leap) works fine
                                    # Leap -> Non-Leap: Drop Feb 29
                                    # Non-Leap -> Leap: Feb 28 fills Feb 29? Valid concern but minor for P50 estimate
                                    
                                    if target_year is not None and target_year != val_year:
                                        # Function to replace year safely
                                        def replace_year(ts):
                                            try:
                                                return ts.replace(year=val_year)
                                            except ValueError:
                                                # Handle Feb 29 in non-leap target year
                                                return ts + pd.DateOffset(days=1) # Shift to Mar 1
                                        
                                        pdf['Time_Central'] = pdf['Time_Source'].apply(replace_year)
                                    else:
                                        pdf['Time_Central'] = pc.index

                                    # 3. Add Energy Calculation immediately
                                    pdf['Gen_Energy_MWh'] = pdf['Gen_MW'] * 0.25
                                    
                                    # 4. Merge with Market Data on Time_Central
                                    # Market Data has 'Time_Central' (US/Central)
                                    # pdf has 'Time_Central' (US/Central) with target year
                                    
                                    merged = pd.merge(df_market_hub, pdf[['Time_Central', 'Gen_Energy_MWh', 'Gen_MW']], on='Time_Central', how='inner')
                                    
                                    # Filter by selected months (use sb_months sidebar multiselect)
                                    m_list = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
                                    selected_month_names = st.session_state.get("sb_months", m_list)
                                    sel_m_nums = [i+1 for i, m in enumerate(m_list) if m in selected_month_names]
                                    if not sel_m_nums:
                                        sel_m_nums = list(range(1, 13))  # fallback: all months
                                    
                                    if sel_m_nums:
                                        merged = merged[merged['Time_Central'].dt.month.isin(sel_m_nums)].copy()

                                    # Settlement Invoice uses bill floating leg (floored at $0) for settlement math.
                                    merged['Settlement_Reference_Price'] = merged['SPP']
                                    if is_bill and invoice_price_series is not None and not merged.empty:
                                        invoice_price_df = pd.DataFrame({
                                            'Time_Central': invoice_price_series.index.tz_convert('US/Central'),
                                            'Invoice_SPP': invoice_price_series.values,
                                        }).drop_duplicates('Time_Central')
                                        merged = pd.merge(merged, invoice_price_df, on='Time_Central', how='left')
                                        merged['Settlement_Reference_Price'] = pd.to_numeric(
                                            merged['Invoice_SPP'],
                                            errors='coerce',
                                        ).fillna(merged['SPP']).clip(lower=0.0)

                                    if preview_tech == "Wind" and not use_sced and not is_bill and not merged.empty:
                                        modeled_mw = apply_congestion_haircut(
                                            gen_series=pd.Series(merged["Gen_MW"].values, index=merged.index),
                                            spp_series=merged["SPP"],
                                            hub_name=calc_hub,
                                            resource_id=selected_project_meta.get("resource_name") if val_source == "Specific Project" else None,
                                        )
                                        merged["Gen_MW"] = modeled_mw.values
                                        merged["Gen_Energy_MWh"] = merged["Gen_MW"] * 0.25

                                    if (
                                        preview_tech == "Wind"
                                        and not use_sced
                                        and not is_bill
                                        and not merged.empty
                                        and not sced_basepoint.empty
                                    ):
                                        merged = pd.merge(
                                            merged,
                                            sced_basepoint,
                                            on="Time_Central",
                                            how="left",
                                        )
                                        if "Base_Point_MW" in merged.columns:
                                            merged["Gen_MW_Raw"] = merged["Gen_MW"]
                                            if preview_apply_bp_cap:
                                                merged["Gen_MW"] = apply_base_point_cap(
                                                    modeled_mw=merged["Gen_MW"],
                                                    base_point_mw=merged["Base_Point_MW"],
                                                    headroom_factor=bp_headroom_factor,
                                                    capacity_mw=preview_capacity,
                                                    cap_strength=DEFAULT_BASE_POINT_CAP_STRENGTH,
                                                )
                                            merged["Base_Point_Headroom_Factor"] = float(bp_headroom_factor)
                                            merged["Base_Point_Cap_Strength"] = (
                                                float(DEFAULT_BASE_POINT_CAP_STRENGTH) if preview_apply_bp_cap else 0.0
                                            )
                                            merged["Base_Point_Cap_Applied"] = bool(preview_apply_bp_cap)
                                            merged["Gen_Energy_MWh"] = merged["Gen_MW"] * 0.25
                                    
                                    # Calculate Potential Curtailment (Always)
                                    merged['Potential_Curtailed_MWh'] = 0.0
                                    mask_neg_price = merged['SPP'] < 0
                                    if not merged.empty:
                                        merged.loc[mask_neg_price, 'Potential_Curtailed_MWh'] = merged.loc[mask_neg_price, 'Gen_Energy_MWh']

                                    merged['Curtailed_MWh'] = 0.0
                                    if not merged.empty:
                                        apply_curtail_to_source = bool(curtail_neg) and (not is_bill)
                                        # Apply Curtailment if selected
                                        if apply_curtail_to_source:
                                            merged.loc[mask_neg_price, 'Curtailed_MWh'] = merged.loc[mask_neg_price, 'Gen_Energy_MWh']
                                            merged.loc[mask_neg_price, 'Gen_Energy_MWh'] = 0
                                        else:
                                            # If not curtailed, 'Curtailed_MWh' is 0, but we have 'Potential' stored
                                            pass
                                        
                                        rs_pct = val_revenue_share / 100.0
                                        
                                        # Apply Negative Price Floor (from UI options)
                                        apply_floor = st.session_state.get("val_use_price_floor", False)
                                        floor_price = st.session_state.get("val_price_floor", -3.0)
                                        
                                        if apply_floor:
                                            effective_spp = np.maximum(merged['Settlement_Reference_Price'], floor_price)
                                        else:
                                            effective_spp = merged['Settlement_Reference_Price']
                                            
                                        if rs_pct < 1.0:
                                            upside = np.maximum(effective_spp - val_vppa_price, 0)
                                            downside = np.minimum(effective_spp - val_vppa_price, 0)
                                            settle_p = (upside * rs_pct) + downside
                                        else:
                                            settle_p = effective_spp - val_vppa_price
                                        merged['Settlement_$/MWh'] = settle_p
                                        merged['Settlement_$'] = merged['Gen_Energy_MWh'] * settle_p
                                        # Uniform settlement uses raw hub SPP (no clip) so all sources are
                                        # on the same price basis for cross-source correlation analysis.
                                        settle_p_uniform = merged['SPP'] - val_vppa_price
                                        merged['Settlement_$/MWh_Uniform'] = settle_p_uniform
                                        merged['Settlement_$_Uniform'] = merged['Gen_Energy_MWh'] * settle_p_uniform
                                        merged['Market_Revenue_$'] = merged['Gen_Energy_MWh'] * merged['Settlement_Reference_Price']
                                        merged['VPPA_Payment_$'] = merged['Gen_Energy_MWh'] * val_vppa_price
                                        merged['VPPA_Price'] = val_vppa_price
                                        preview_results[source["name"]] = merged
                            
                            if preview_results:
                                st.session_state['val_preview_results'] = preview_results
                                st.session_state['val_preview_tech'] = preview_tech
                            else:
                                st.error("No data generated for criteria.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Month selection is now in the config row above (multiselect).


    st.caption("💡 Enter your project's exact coordinates or use the map below")
    
    # --- Layout Containers (Visual Order) ---
    input_container = st.container()
    map_container = st.expander("🗺️ Pick Project Location", expanded=False)
    
    # --- Map Logic (Execution Order: Run First) ---
    # We run this BEFORE inputs so that click updates (session_state) are ready for the input widgets
    with map_container:
        st.caption("Search by name or click on the map to select your project location")
        
        # Location search box
        val_search_query = st.text_input("🔍 Search location", placeholder="e.g., Abilene, TX or 79601", key="val_location_search")
        
        if val_search_query:
            try:
                geolocator = Nominatim(user_agent="vppa_estimator")
                # Append Texas to improve search accuracy
                if "texas" not in val_search_query.lower() and "tx" not in val_search_query.lower():
                    search_with_state = f"{val_search_query}, Texas, USA"
                else:
                    search_with_state = f"{val_search_query}, USA"
                
                location = geolocator.geocode(search_with_state, timeout=5)
                
                if location:
                    # Clamp to Texas bounds
                    found_lat = max(25.5, min(36.5, location.latitude))
                    found_lon = max(-106.5, min(-93.5, location.longitude))
                    
                    st.session_state.val_map_lat = found_lat
                    st.session_state.val_map_lon = found_lon
                    st.session_state.val_custom_lat = found_lat
                    st.session_state.val_custom_lon = found_lon
                    # Auto-check the "Use Custom Location" checkbox (Safe here because widget not yet created)
                    st.session_state.val_use_custom_location = True
                    st.success(f"📍 Found: {location.address[:50]}...")
                    st.caption(f"Coordinates: {found_lat:.4f}, {found_lon:.4f}")
                else:
                    st.warning("Location not found. Try a different name.")
            except GeocoderTimedOut:
                st.warning("Search timed out. Try again.")
            except Exception as e:
                st.error(f"Search error: {str(e)[:50]}")
        
        # Initialize map location from session state or defaults
        if 'val_map_lat' not in st.session_state:
            st.session_state.val_map_lat = 32.0
        if 'val_map_lon' not in st.session_state:
            st.session_state.val_map_lon = -100.0
        
        # Sync map with custom location inputs if manually entered
        if 'val_custom_lat' in st.session_state and 'val_custom_lon' in st.session_state:
            st.session_state.val_map_lat = st.session_state.val_custom_lat
            st.session_state.val_map_lon = st.session_state.val_custom_lon
        
        # Create map centered on selected location (not just Texas center)
        val_map = folium.Map(
            location=[st.session_state.val_map_lat, st.session_state.val_map_lon],  # Center on selected location
            zoom_start=7,
            tiles="OpenStreetMap"
        )
        
        # Add marker for current selected location
        folium.Marker(
            [st.session_state.val_map_lat, st.session_state.val_map_lon],
            popup=f"Selected: {st.session_state.val_map_lat:.4f}, {st.session_state.val_map_lon:.4f}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(val_map)
        
        # Add ERCOT hub markers for reference
        for hub, (lat, lon) in HUB_LOCATIONS.items():
            folium.CircleMarker(
                [lat, lon],
                radius=10,
                popup=f"Hub: {hub}",
                color='blue',
                fill=True,
                fillOpacity=0.6,
                weight=2
            ).add_to(val_map)
            
        # Add Real Asset markers
        try:
            with open('ercot_assets.json', 'r') as f:
                asset_registry = json.load(f)
        except Exception:
            asset_registry = {}
            
        for name, meta in asset_registry.items():
            icon_color = 'orange' if meta['tech'] == 'Solar' else 'blue'
            icon_name = 'sun' if meta['tech'] == 'Solar' else 'cloud'
            
            folium.Marker(
                [meta['lat'], meta['lon']],
                popup=f"<b>Asset: {name}</b><br>Type: {meta['tech']}<br>Capacity: {meta['capacity_mw']} MW",
                tooltip=f"{name} ({meta['tech']})",
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')
            ).add_to(val_map)

        
        # Display map and capture clicks
        val_map_data = st_folium(val_map, height=400, width=None, returned_objects=["last_clicked"], key="val_map")
        
        if val_map_data and val_map_data.get("last_clicked"):
            clicked_lat = val_map_data["last_clicked"]["lat"]
            clicked_lon = val_map_data["last_clicked"]["lng"]
            # Clamp to Texas bounds to prevent errors in form inputs
            clicked_lat = max(25.5, min(36.5, clicked_lat))
            clicked_lon = max(-106.5, min(-93.5, clicked_lon))
            st.session_state.val_map_lat = clicked_lat
            st.session_state.val_map_lon = clicked_lon
            
            # Check if we need to update the inputs
            current_lat = st.session_state.get('val_custom_lat', 0.0)
            current_lon = st.session_state.get('val_custom_lon', 0.0)
            
            if abs(current_lat - clicked_lat) > 0.0001 or abs(current_lon - clicked_lon) > 0.0001:
                # Sync inputs - THIS WORKS safely because inputs are rendered AFTER this block
                st.session_state.val_custom_lat = clicked_lat
                st.session_state.val_custom_lon = clicked_lon
                st.session_state.val_use_custom_location = True
                st.rerun()


    # Map Logic continues above...
    
    # Custom Location Toggle and Manual Input (Moved after Map Logic)
    # --- Input Widgets (Visual Order: Top, Execution Order: Second) ---
    with input_container:
        val_use_custom_location = st.checkbox("Use Custom Project Location", value=False, help="Specify exact project coordinates", key="val_use_custom_location")
    
        # Auto-populate defaults when switching to Custom mode
        if val_use_custom_location and not st.session_state.get('prev_use_custom', False):
            # 33° 9' 12.2" N, 99° 17' 4.8" W
            st.session_state.val_custom_lat = 33.1534
            st.session_state.val_custom_lon = -99.2847
            st.session_state.prev_use_custom = True
            st.rerun()
    
        st.session_state.prev_use_custom = val_use_custom_location

        col_lat, col_lon = st.columns(2)
        with col_lat:
            val_custom_lat = st.number_input("Latitude", min_value=25.0, max_value=40.0, step=0.01, format="%.4f", key="val_custom_lat")
        with col_lon:
            val_custom_lon = st.number_input("Longitude", min_value=-107.0, max_value=-93.0, step=0.01, format="%.4f", key="val_custom_lon")
    
    st.info(f"📍 Selected location: {val_custom_lat:.4f}, {val_custom_lon:.4f}")
    
    if st.session_state.get('val_use_custom_location', False):
         st.success(f"✅ Using custom location: {val_custom_lat:.4f}, {val_custom_lon:.4f}")
    


    
    # --- Display Results Section (Cached) ---
    if 'val_preview_results' in st.session_state:
        preview_results = st.session_state['val_preview_results']
        
        # 1. Comparison Table if multiple sources
        if len(preview_results) > 1:
            # Dynamic title based on weather source
            if preview_weather == "Model Based on Weather":
                comparison_title = "### 📊 Contrast: Actual SCED vs Model"
            elif preview_weather == "Actual SCED + Settlement Invoice":
                comparison_title = "### 📊 Contrast: Actual SCED vs Settlement Invoice"
            elif preview_weather == "Actual SCED + Model Based on Weather + Settlement Invoice":
                comparison_title = "### 📊 Contrast: SCED vs Model vs Settlement Invoice"
            else:
                comparison_title = "### 📊 Contrast: Actual vs Typical (TMY)"
            
            st.markdown(comparison_title)
            comp_summary = []
            comp_monthly_metrics = []
            for name, df in preview_results.items():
                t_gen = df['Gen_Energy_MWh'].sum()
                t_settle = df['Settlement_$'].sum()
                t_rev = df['Market_Revenue_$'].sum()
                
                c_price = t_rev / t_gen if t_gen > 0 else 0
                rec_cost = -(t_settle / t_gen) if t_gen > 0 else 0
                
                comp_summary.append({
                    "Source": name,
                    "Generation (MWh)": f"{t_gen:,.0f}",
                    "Capture Price ($/MWh)": f"${c_price:.2f}",
                    "Implied REC Cost ($/MWh)": f"${rec_cost:.2f}",
                    "Net Settlement ($)": f"${t_settle:,.0f}"
                })
                
                # Monthly breakout for the same summary metrics.
                if not df.empty and 'Time_Central' in df.columns:
                    df_month = df.copy()
                    df_month['MonthPeriod'] = pd.to_datetime(df_month['Time_Central'], errors='coerce').dt.to_period('M')
                    df_month = df_month.dropna(subset=['MonthPeriod'])
                    # Filter to only selected months (same as computation filter)
                    _m_list = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
                    _sel_names = st.session_state.get("sb_months", _m_list)
                    _sel_nums = [i+1 for i, m in enumerate(_m_list) if m in _sel_names] or list(range(1, 13))
                    df_month = df_month[pd.to_datetime(df_month['Time_Central'], errors='coerce').dt.month.isin(_sel_nums)]
                    if not df_month.empty:
                        month_agg = (
                            df_month.groupby('MonthPeriod', as_index=False)[['Gen_Energy_MWh', 'Settlement_$', 'Market_Revenue_$']]
                            .sum()
                        )
                        for _, r in month_agg.iterrows():
                            m_gen = float(r['Gen_Energy_MWh'])
                            m_settle = float(r['Settlement_$'])
                            m_rev = float(r['Market_Revenue_$'])
                            m_capture = (m_rev / m_gen) if m_gen > 0 else 0.0
                            m_rec_cost = -(m_settle / m_gen) if m_gen > 0 else 0.0
                            comp_monthly_metrics.append({
                                "MonthPeriod": pd.Period(str(r['MonthPeriod']), freq='M'),
                                "Month": pd.Period(str(r['MonthPeriod']), freq='M').strftime('%b %Y'),
                                "Source": name,
                                "Generation (MWh)": m_gen,
                                "Capture Price ($/MWh)": m_capture,
                                "Implied REC Cost ($/MWh)": m_rec_cost,
                                "Net Settlement ($)": m_settle,
                            })
            st.table(pd.DataFrame(comp_summary))
            if comp_monthly_metrics:
                st.markdown("#### By Month")
                df_comp_monthly = pd.DataFrame(comp_monthly_metrics).sort_values(
                    by=["MonthPeriod", "Source"]
                )

                if {"SCED_Actual", "Model"}.issubset(set(df_comp_monthly["Source"].unique())):
                    df_sced = (
                        df_comp_monthly[df_comp_monthly["Source"] == "SCED_Actual"]
                        .set_index("MonthPeriod")
                        [["Month", "Generation (MWh)", "Capture Price ($/MWh)", "Implied REC Cost ($/MWh)", "Net Settlement ($)"]]
                    )
                    df_model = (
                        df_comp_monthly[df_comp_monthly["Source"] == "Model"]
                        .set_index("MonthPeriod")
                        [["Generation (MWh)", "Capture Price ($/MWh)", "Implied REC Cost ($/MWh)", "Net Settlement ($)"]]
                    )
                    df_month_cmp = df_sced.join(
                        df_model,
                        how="inner",
                        lsuffix=" SCED",
                        rsuffix=" Model",
                    ).reset_index(drop=True)

                    df_month_cmp["Generation Δ (MWh)"] = (
                        df_month_cmp["Generation (MWh) Model"] - df_month_cmp["Generation (MWh) SCED"]
                    )
                    df_month_cmp["Generation Δ (%)"] = np.where(
                        df_month_cmp["Generation (MWh) SCED"] != 0,
                        (df_month_cmp["Generation Δ (MWh)"] / df_month_cmp["Generation (MWh) SCED"]) * 100.0,
                        np.nan,
                    )
                    df_month_cmp["Capture Δ ($/MWh)"] = (
                        df_month_cmp["Capture Price ($/MWh) Model"] - df_month_cmp["Capture Price ($/MWh) SCED"]
                    )
                    df_month_cmp["REC Δ ($/MWh)"] = (
                        df_month_cmp["Implied REC Cost ($/MWh) Model"] - df_month_cmp["Implied REC Cost ($/MWh) SCED"]
                    )
                    df_month_cmp["Net Settlement Δ ($)"] = (
                        df_month_cmp["Net Settlement ($) Model"] - df_month_cmp["Net Settlement ($) SCED"]
                    )
                    df_month_cmp["Net Settlement Δ (%)"] = np.where(
                        df_month_cmp["Net Settlement ($) SCED"] != 0,
                        (df_month_cmp["Net Settlement Δ ($)"] / df_month_cmp["Net Settlement ($) SCED"]) * 100.0,
                        np.nan,
                    )

                    display_cols = [
                        "Month",
                        "Generation (MWh) SCED",
                        "Generation (MWh) Model",
                        "Generation Δ (MWh)",
                        "Generation Δ (%)",
                        "Capture Price ($/MWh) SCED",
                        "Capture Price ($/MWh) Model",
                        "Capture Δ ($/MWh)",
                        "Implied REC Cost ($/MWh) SCED",
                        "Implied REC Cost ($/MWh) Model",
                        "REC Δ ($/MWh)",
                        "Net Settlement ($) SCED",
                        "Net Settlement ($) Model",
                        "Net Settlement Δ ($)",
                        "Net Settlement Δ (%)",
                    ]

                    st.dataframe(
                        df_month_cmp[display_cols].style.format(
                            {
                                "Generation (MWh) SCED": "{:,.0f}",
                                "Generation (MWh) Model": "{:,.0f}",
                                "Generation Δ (MWh)": "{:+,.0f}",
                                "Generation Δ (%)": "{:+.2f}%",
                                "Capture Price ($/MWh) SCED": "${:.2f}",
                                "Capture Price ($/MWh) Model": "${:.2f}",
                                "Capture Δ ($/MWh)": "{:+.2f}",
                                "Implied REC Cost ($/MWh) SCED": "${:.2f}",
                                "Implied REC Cost ($/MWh) Model": "${:.2f}",
                                "REC Δ ($/MWh)": "{:+.2f}",
                                "Net Settlement ($) SCED": "${:,.0f}",
                                "Net Settlement ($) Model": "${:,.0f}",
                                "Net Settlement Δ ($)": "{:+,.0f}",
                                "Net Settlement Δ (%)": "{:+.2f}%",
                            }
                        ),
                        use_container_width=True,
                    )
                else:
                    st.dataframe(
                        df_comp_monthly.drop(columns=["MonthPeriod"]).style.format(
                            {
                                "Generation (MWh)": "{:,.0f}",
                                "Capture Price ($/MWh)": "${:.2f}",
                                "Implied REC Cost ($/MWh)": "${:.2f}",
                                "Net Settlement ($)": "${:,.0f}",
                            }
                        ),
                        use_container_width=True,
                    )
            st.markdown("---")
        
        # 2. Main Metrics
        source_names = list(preview_results.keys())
        preferred_source_order = ["SCED_Actual", "Actual", "Settlement_Invoice", "Model", "TMY", "P50"]
        primary_name = next((s for s in preferred_source_order if s in source_names), source_names[0])
        df_primary = preview_results[primary_name]
        
        total_gen = df_primary['Gen_Energy_MWh'].sum()
        
        # Determine what to show for curtailment metric
        # Use Potential if it exists, otherwise fallback
        pot_curtailed = df_primary['Potential_Curtailed_MWh'].sum() if 'Potential_Curtailed_MWh' in df_primary.columns else 0
        act_curtailed = df_primary['Curtailed_MWh'].sum() if 'Curtailed_MWh' in df_primary.columns else 0
        
        curtail_metric_val = act_curtailed if curtail_neg else pot_curtailed
        curtail_metric_label = "Curtailed Gen" if curtail_neg else "Neg. Price Gen"
        curtail_metric_help = "Actual curtailed MWh" if curtail_neg else "MWh generated during negative prices (Potential Curtailment)"

        total_settlement = df_primary['Settlement_$'].sum()
        total_market_revenue = df_primary['Market_Revenue_$'].sum()
        
        # Calculate Split Breakdown
        total_received = df_primary['Market_Revenue_$'].sum()
        total_paid = df_primary['VPPA_Payment_$'].sum()

        avg_spp = df_primary['SPP'].mean()
        capture_price = total_market_revenue / total_gen if total_gen > 0 else 0
        implied_rec_cost = -(total_settlement / total_gen) if total_gen > 0 else 0
        
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        col1.metric("Total Generation", f"{total_gen:,.0f} MWh")
        col2.metric(curtail_metric_label, f"{curtail_metric_val:,.0f} MWh", help=curtail_metric_help)
        col3.metric("Total Settlement", f"${total_settlement:,.0f}")
        col4.metric("Total Paid", f"${total_paid:,.0f}", help="Total fixed amount paid (Generation × VPPA Price)")
        col5.metric("Total Received", f"${total_received:,.0f}", help="Total market revenue received (Generation × Market Price)")
        col6.metric("Avg Hub Price", f"${avg_spp:.2f}/MWh")
        col7.metric("Capture Price", f"${capture_price:.2f}/MWh", 
                  help="Weighted average market value of generated energy (Market Revenue / Generation)")
        col8.metric("Implied REC Cost", f"${implied_rec_cost:.2f}/MWh",
                   help="Net cost (positive) or credit (negative) paid due to PPA settlement. Calculated as -(Settlement / Generation).")

        # Monthly KPI table for all available sources at once.
        monthly_curtail_col = "Curtailed_MWh" if curtail_neg else "Potential_Curtailed_MWh"
        monthly_curtail_label = "Curtailed Gen (MWh)" if curtail_neg else "Neg. Price Gen (MWh)"
        monthly_rows = []

        for source_name, df_source in preview_results.items():
            if df_source.empty or "Time_Central" not in df_source.columns:
                continue

            df_monthly_kpi = df_source.copy()
            df_monthly_kpi["MonthPeriod"] = pd.to_datetime(df_monthly_kpi["Time_Central"], errors="coerce").dt.to_period("M")
            df_monthly_kpi = df_monthly_kpi.dropna(subset=["MonthPeriod"])
            # Filter to only user-selected months
            if selected_month_numbers:
                df_monthly_kpi = df_monthly_kpi[
                    pd.to_datetime(df_monthly_kpi["Time_Central"], errors="coerce").dt.month.isin(selected_month_numbers)
                ]
            if df_monthly_kpi.empty:
                continue

            # Ensure expected columns exist for consistent aggregation across sources.
            if "Gen_Energy_MWh" not in df_monthly_kpi.columns:
                df_monthly_kpi["Gen_Energy_MWh"] = 0.0
            if "Curtailed_MWh" not in df_monthly_kpi.columns:
                df_monthly_kpi["Curtailed_MWh"] = 0.0
            if "Potential_Curtailed_MWh" not in df_monthly_kpi.columns:
                df_monthly_kpi["Potential_Curtailed_MWh"] = 0.0
            if "Settlement_$" not in df_monthly_kpi.columns:
                df_monthly_kpi["Settlement_$"] = 0.0
            if "VPPA_Payment_$" not in df_monthly_kpi.columns:
                df_monthly_kpi["VPPA_Payment_$"] = 0.0
            if "Market_Revenue_$" not in df_monthly_kpi.columns:
                df_monthly_kpi["Market_Revenue_$"] = 0.0
            if "SPP" not in df_monthly_kpi.columns:
                df_monthly_kpi["SPP"] = np.nan

            month_kpi = (
                df_monthly_kpi.groupby("MonthPeriod", as_index=False)
                .agg(
                    {
                        "Gen_Energy_MWh": "sum",
                        "Curtailed_MWh": "sum",
                        "Potential_Curtailed_MWh": "sum",
                        "Settlement_$": "sum",
                        "VPPA_Payment_$": "sum",
                        "Market_Revenue_$": "sum",
                        "SPP": "mean",
                    }
                )
                .sort_values("MonthPeriod")
            )

            month_kpi["Month"] = month_kpi["MonthPeriod"].dt.strftime("%b %Y")
            month_kpi["Source"] = source_name
            month_kpi["Capture Price ($/MWh)"] = np.where(
                month_kpi["Gen_Energy_MWh"] > 0,
                month_kpi["Market_Revenue_$"] / month_kpi["Gen_Energy_MWh"],
                0.0,
            )
            month_kpi["Implied REC Cost ($/MWh)"] = np.where(
                month_kpi["Gen_Energy_MWh"] > 0,
                -(month_kpi["Settlement_$"] / month_kpi["Gen_Energy_MWh"]),
                0.0,
            )
            month_kpi = month_kpi.rename(
                columns={
                    "Gen_Energy_MWh": "Total Generation (MWh)",
                    monthly_curtail_col: monthly_curtail_label,
                    "Settlement_$": "Total Settlement ($)",
                    "VPPA_Payment_$": "Total Paid ($)",
                    "Market_Revenue_$": "Total Received ($)",
                    "SPP": "Avg Hub Price ($/MWh)",
                }
            )
            monthly_rows.append(month_kpi)

        if monthly_rows:
            month_kpi_all = pd.concat(monthly_rows, ignore_index=True).sort_values(["MonthPeriod", "Source"])
            display_cols = [
                "Month",
                "Source",
                "Total Generation (MWh)",
                monthly_curtail_label,
                "Total Settlement ($)",
                "Total Paid ($)",
                "Total Received ($)",
                "Avg Hub Price ($/MWh)",
                "Capture Price ($/MWh)",
                "Implied REC Cost ($/MWh)",
            ]

            st.markdown("#### Monthly KPIs (All Sources)")
            st.dataframe(
                month_kpi_all[display_cols].style.format(
                    {
                        "Total Generation (MWh)": "{:,.0f}",
                        monthly_curtail_label: "{:,.0f}",
                        "Total Settlement ($)": "${:,.0f}",
                        "Total Paid ($)": "${:,.0f}",
                        "Total Received ($)": "${:,.0f}",
                        "Avg Hub Price ($/MWh)": "${:.2f}",
                        "Capture Price ($/MWh)": "${:.2f}",
                        "Implied REC Cost ($/MWh)": "${:.2f}",
                    },
                    na_rep="-",
                ),
                use_container_width=True,
            )

        corr_df = build_multi_source_correlation_analysis(preview_results, selected_month_numbers)
        if not corr_df.empty:
            context_label = selected_project_name if (val_source == "Specific Project" and selected_project_name) else val_hub
            st.markdown("#### Multi-Source Correlation Analysis")
            st.caption(f"{context_label} {val_year} - Multi-Source Correlation Analysis")
            st.caption("Source color key: Purple = SCED Actual vs Model | Orange = SCED Actual vs Invoice | Green = Model vs Invoice")
            st.dataframe(
                corr_df.style.format("{:.4f}", na_rep="-"),
                use_container_width=True,
            )
            corr_by_month = build_multi_source_correlation_by_month(preview_results, selected_month_numbers)
            if corr_by_month:
                st.markdown("##### Correlations by Month")
                month_labels = list(corr_by_month.keys())
                month_tabs = st.tabs(month_labels)
                for tab, month_label in zip(month_tabs, month_labels):
                    with tab:
                        st.dataframe(
                            corr_by_month[month_label].style.format("{:.4f}", na_rep="-"),
                            use_container_width=True,
                        )

        st.markdown("### 📤 Export Monthly Comparison Report")
        st.caption("Exports a formatted Excel report for the selected months and currently active sources (SCED / Model / Invoice).")
        try:
            report_context_label = selected_project_name if (val_source == "Specific Project" and selected_project_name) else val_hub
            report_bytes = build_monthly_comparison_report_excel(
                preview_results=preview_results,
                selected_month_numbers=selected_month_numbers,
                val_year=val_year,
                selected_project_name=selected_project_name,
                selected_resource_id=selected_resource_id,
                report_context_label=report_context_label,
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_stub = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(report_context_label or "monthly_comparison")).strip("_")
            file_stub = file_stub or "monthly_comparison"
            st.download_button(
                label="📥 Download Monthly Comparison Report (Excel)",
                data=report_bytes,
                file_name=f"{file_stub}_monthly_comparison_{val_year}_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_monthly_comparison_report_xlsx",
            )
        except Exception as e:
            st.warning(f"Could not build monthly comparison report: {e}")
        
        # 3. Charts with own View Selector
        st.markdown("---")
        chart_col, view_col = st.columns([0.7, 0.3])
        with chart_col:
            st.markdown("### 📊 Settlement Chart")
        with view_col:
            preview_view = st.selectbox("Chart Time Aggregation", ["Daily", "Monthly"], key="preview_view_internal")

        fig = go.Figure()
        
        for name, df in preview_results.items():
            if selected_month_numbers:
                df = df[df['Time_Central'].dt.month.isin(selected_month_numbers)]
            if preview_view == "Monthly":
                df['Month'] = df['Time_Central'].dt.to_period('M').astype(str)
                chart_df = df.groupby('Month').agg({'Settlement_$': 'sum'}).reset_index()
                x_col = 'Month'
            else: # Daily
                df['Date'] = df['Time_Central'].dt.date
                chart_df = df.groupby('Date').agg({'Settlement_$': 'sum'}).reset_index()
                x_col = 'Date'

            if len(preview_results) > 1:
                if preview_view == "Monthly":
                    # Use grouped bars for monthly comparison
                    fig.add_trace(go.Bar(
                        x=chart_df['Month'],
                        y=chart_df['Settlement_$'],
                        name=f'{name} Settlement'
                    ))
                else:
                    # Use Line when comparing daily
                    fig.add_trace(go.Scatter(
                        x=chart_df['Date'],
                        y=chart_df['Settlement_$'],
                        name=f'{name} Settlement',
                        mode='lines',
                        opacity=0.8
                    ))
            else:
                # Use Bar for single source
                fig.add_trace(go.Bar(
                    x=chart_df[x_col],
                    y=chart_df['Settlement_$'],
                    name=f'{preview_view} Settlement',
                    marker_color=['green' if x > 0 else 'red' for x in chart_df['Settlement_$']]
                ))
                
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        chart_title = f"{preview_view} Net Settlement"
        if len(preview_results) > 1:
            chart_title += " Comparison"
            if preview_view == "Monthly":
                fig.update_layout(barmode='group')
                
        fig.update_layout(
            title=chart_title,
            xaxis_title="Month" if preview_view == "Monthly" else "Date",
            yaxis_title="Settlement ($)",
            hovermode="x unified",
            height=450
        )
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)
        
        # 3b. Generation Chart
        st.markdown("---")
        st.markdown("### ⚡ Generation Chart")
        
        fig_gen = go.Figure()
        
        for name, df in preview_results.items():
            if selected_month_numbers:
                df = df[df['Time_Central'].dt.month.isin(selected_month_numbers)]
            if preview_view == "Monthly":
                # derivating Month aggregator
                df['Month'] = df['Time_Central'].dt.to_period('M').astype(str)
                gen_df = df.groupby('Month').agg({'Gen_Energy_MWh': 'sum'}).reset_index()
                x_col = 'Month'
            else: # Daily
                df['Date'] = df['Time_Central'].dt.date
                gen_df = df.groupby('Date').agg({'Gen_Energy_MWh': 'sum'}).reset_index()
                x_col = 'Date'
            
            if len(preview_results) > 1:
                if preview_view == "Monthly":
                    # Use grouped bars for monthly comparison
                    fig_gen.add_trace(go.Bar(
                        x=gen_df['Month'],
                        y=gen_df['Gen_Energy_MWh'],
                        name=f'{name} Generation'
                    ))
                else:
                    # Use Line when comparing daily
                    fig_gen.add_trace(go.Scatter(
                        x=gen_df['Date'],
                        y=gen_df['Gen_Energy_MWh'],
                        name=f'{name} Generation',
                        mode='lines',
                        opacity=0.8
                    ))
            else:
                # Use Bar for single source
                fig_gen.add_trace(go.Bar(
                    x=gen_df[x_col],
                    y=gen_df['Gen_Energy_MWh'],
                    name=f'{preview_view} Generation',
                    marker_color='#1f77b4' # Muted blue
                ))
                
        gen_title = f"{preview_view} Renewable Generation"
        if len(preview_results) > 1:
            gen_title += " Comparison"
            if preview_view == "Monthly":
                fig_gen.update_layout(barmode='group')
                
        fig_gen.update_layout(
            title=gen_title,
            xaxis_title="Month" if preview_view == "Monthly" else "Date",
            yaxis_title="Generation (MWh)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450
        )
        st.plotly_chart(fig_gen, use_container_width=True)
        
        # 4. Preview and Download for Primary
        st.markdown(f"### 📋 15-Minute Interval Data Preview ({primary_name})")
        st.caption("Showing first 100 intervals")

        df_primary_filtered = df_primary.copy()
        if selected_month_numbers:
            df_primary_filtered = df_primary_filtered[df_primary_filtered['Time_Central'].dt.month.isin(selected_month_numbers)]

        df_primary_filtered['Implied_REC_Cost_$'] = df_primary_filtered['Settlement_$']

        if 'VPPA_Price' not in df_primary_filtered.columns:
            df_primary_filtered['VPPA_Price'] = val_vppa_price

        display_cols = ['Time_Central', 'Gen_MW', 'Gen_Energy_MWh', 'SPP', 'Settlement_$/MWh', 'Settlement_$', 'Implied_REC_Cost_$', 'VPPA_Price']
        preview_df = df_primary_filtered[display_cols].head(100).copy()
        preview_df['Time_Central'] = preview_df['Time_Central'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(
            preview_df.style.format({
                'Gen_MW': '{:.2f}',
                'Gen_Energy_MWh': '{:.4f}',
                'SPP': '${:.2f}',
                'Settlement_$/MWh': '${:.2f}',
                'Settlement_$': '${:.2f}',
                'Implied_REC_Cost_$': '${:.2f}',
                'VPPA_Price': '${:.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Download full dataset
        csv = df_primary_filtered[['Time_Central'] + display_cols[1:]].to_csv(index=False).encode('utf-8')
        p_tech = st.session_state.get('val_preview_tech', 'profile')
        st.download_button(
            label=f"📥 Download Full {primary_name} Intervals CSV",
            data=csv,
            file_name=f"{p_tech}_{st.session_state.get('val_hub', 'HUB')}_{val_year}_{primary_name}_intervals.csv",
            mime="text/csv",
            key="download_preview_csv"
        )

        # Download Combined Scenarios (if multiple)
        if len(preview_results) > 1:
            st.divider()
            st.markdown("### 📥 Download Comparison Data")
            st.caption(f"Includes columns for: {', '.join(preview_results.keys())}")

            combined_df = None
            base_cols = ['Time_Central', 'Gen_MW', 'Gen_Energy_MWh', 'SPP', 'Settlement_$', 'Settlement_$/MWh']
            
            for name, df_scen in preview_results.items():
                if selected_month_numbers:
                    df_scen = df_scen[df_scen['Time_Central'].dt.month.isin(selected_month_numbers)]
                # Ensure Implied REC Cost is calculated
                if 'Implied_REC_Cost_$' not in df_scen.columns:
                     df_scen['Implied_REC_Cost_$'] = df_scen['Settlement_$']

                if 'VPPA_Price' not in df_scen.columns:
                     df_scen['VPPA_Price'] = val_vppa_price

                # Select columns
                cols_to_use = base_cols + ['Implied_REC_Cost_$', 'VPPA_Price']

                temp_df = df_scen[cols_to_use].copy()
                
                # Rename columns (except Time)
                temp_df.columns = ['Time_Central'] + [f"{c}_{name}" for c in cols_to_use[1:]]
                
                if combined_df is None:
                    combined_df = temp_df
                else:
                    combined_df = pd.merge(combined_df, temp_df, on='Time_Central', how='outer')
            
            if combined_df is not None:
                combined_df = combined_df.sort_values('Time_Central')
                combined_csv = combined_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Download All Scenarios (Merged CSV)",
                    data=combined_csv,
                    file_name=f"comparison_all_{st.session_state.get('val_hub', 'HUB')}_{val_year}_intervals.csv",
                    mime="text/csv",
                    key="download_all_scenarios_csv"
                )

        # Download PDF Bill
        pdf_config = {
            'hub': st.session_state.get('val_hub', 'HB_NORTH'),
            'year': st.session_state.get('val_year', val_year),
            'tech': st.session_state.get('preview_tech', 'Unknown'),
            'capacity_mw': st.session_state.get('preview_capacity', 100.0),
            'vppa_price': st.session_state.get('val_price', 0.0),
            'location_name': st.session_state.get('val_location_search', '') or 'Project Site',
            'lat': st.session_state.get('val_map_lat', 0.0),
            'lon': st.session_state.get('val_map_lon', 0.0)
        }
        
        pdf_data = generate_settlement_pdf(df_primary, pdf_config)
        
        st.download_button(
            label="📄 Download Settlement Bill (PDF)",
            data=pdf_data,
            file_name=f"Settlement_Bill_{st.session_state.get('val_hub', 'HUB')}_{st.session_state.get('val_year', 'YEAR')}.pdf",
            mime="application/pdf",
            key="download_bill_pdf_final"
        )
        
    st.markdown("---")
    st.subheader("📤 Upload Your Bill for Validation")


    # --- File Uploader ---
    uploaded_bill = st.file_uploader(
        "Upload Generation Data", 
        type=["csv", "xlsx", "xls", "pdf"], 
        help="Supported formats: CSV, Excel (.xlsx/.xls), PDF. Required columns: 'Time' (or 'Date', 'Interval'), 'Generation' (or 'MW', 'Quantity'). Optional: 'Settlement' (for comparison)."
    )

    if uploaded_bill:
        try:
            # 1. Load User Data based on file type
            file_extension = uploaded_bill.name.rsplit('.', 1)[-1].lower()
            if file_extension not in {"csv", "xlsx", "xls", "pdf"}:
                st.error(f"Unsupported file type: {file_extension}")
                st.stop()

            file_bytes = uploaded_bill.getvalue()
            df_bill = parse_uploaded_bill_file(file_bytes, file_extension)
            
            # Normalize Columns
            df_bill.columns = [c.lower().strip() if c else f'col_{i}' for i, c in enumerate(df_bill.columns)]
            
            # Identify Key Columns
            time_col = next((c for c in df_bill.columns if any(x in c for x in ['time', 'date', 'interval', 'hour'])), None)
            gen_col = next((c for c in df_bill.columns if any(x in c for x in ['gen', 'mw', 'quantity', 'mwh'])), None)
            settlement_col = next((c for c in df_bill.columns if any(x in c for x in ['settlement', 'amount', 'revenue', 'value'])), None)
            
            if not time_col or not gen_col:
                st.error(f"Could not identify required columns. Found: {list(df_bill.columns)}. Need Time/Date and Gen/MW.")
            else:
                # 2. Process User Data
                parsed_time = pd.to_datetime(df_bill[time_col], errors='coerce')
                invalid_time_rows = int(parsed_time.isna().sum())
                if invalid_time_rows:
                    df_bill = df_bill.loc[parsed_time.notna()].copy()
                    parsed_time = parsed_time.loc[parsed_time.notna()]
                    st.warning(f"Skipped {invalid_time_rows:,} rows with invalid timestamps.")

                if df_bill.empty:
                    st.error("No valid timestamps were found in the uploaded bill.")
                    st.stop()

                if parsed_time.dt.tz is None:
                    parsed_time = parsed_time.dt.tz_localize('US/Central', ambiguous='NaT', nonexistent='shift_forward')
                    ambiguous_time_rows = int(parsed_time.isna().sum())
                    if ambiguous_time_rows:
                        df_bill = df_bill.loc[parsed_time.notna()].copy()
                        parsed_time = parsed_time.loc[parsed_time.notna()]
                        st.warning(f"Skipped {ambiguous_time_rows:,} rows due to ambiguous DST timestamps.")
                    if df_bill.empty:
                        st.error("No valid timestamps remained after timezone normalization.")
                        st.stop()

                df_bill['Time'] = parsed_time.dt.tz_convert('UTC')
                
                # Resample/Align if needed? For now assume it matches roughly or we align to Market Data
                # Rename for clarity
                df_bill = df_bill.rename(columns={gen_col: 'User_Gen_MW'})
                if settlement_col:
                    df_bill = df_bill.rename(columns={settlement_col: 'User_Settlement_Amount'})
                
                # Convert to numeric (in case PDF/Excel had strings)
                df_bill['User_Gen_MW'] = pd.to_numeric(df_bill['User_Gen_MW'], errors='coerce')
                
                if 'User_Settlement_Amount' in df_bill.columns:
                    # Clean currency formatting if string
                    if df_bill['User_Settlement_Amount'].dtype == 'object':
                        df_bill['User_Settlement_Amount'] = df_bill['User_Settlement_Amount'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace(')', '', regex=False).str.replace('(', '-', regex=False)
                    
                    df_bill['User_Settlement_Amount'] = pd.to_numeric(df_bill['User_Settlement_Amount'], errors='coerce').fillna(0)
                else:
                    # Ensure column exists for aggregation later, defaulting to 0
                    df_bill['User_Settlement_Amount'] = 0.0

                # Reuse Central-time month keys throughout validation to avoid repeated datetime parsing.
                df_bill['Time_Central'] = df_bill['Time'].dt.tz_convert('US/Central')
                df_bill['MonthPeriod'] = df_bill['Time_Central'].dt.to_period('M')
                df_bill['Month'] = df_bill['MonthPeriod'].dt.strftime('%b %Y')

                if selected_month_numbers:
                    df_bill = df_bill[df_bill['Time_Central'].dt.month.isin(selected_month_numbers)].copy()
                    if df_bill.empty:
                        st.error("No uploaded bill rows fall within the selected validation months.")
                        st.stop()
                
                # Detect if this is monthly summary data vs interval data
                # Monthly summary typically has <= 12 rows (one per month)
                is_monthly_summary = len(df_bill) <= 24 and 'User_Settlement_Amount' in df_bill.columns
                
                # --- 3. Model vs. Actual Comparison Logic ---
                # Check if we have modeled results in session state to compare against
                modeled_results = st.session_state.get('val_preview_results')
                
                if modeled_results:
                    st.success("✅ Modeled scenario found! Comparing your Uploaded Bill against the Model.")
                    
                    # Get primary modeled dataframe (Actual or first available)
                    primary_name = "Actual" if "Actual" in modeled_results else list(modeled_results.keys())[0]
                    df_model = modeled_results[primary_name].copy()
                    
                    # Align Data for Comparison
                    # We need to aggregate both to the same granularity (Monthly or Totals) for high-level variance
                    
                    # Aggregate Model to Monthly
                    df_model['MonthPeriod'] = df_model['Time_Central'].dt.to_period('M')
                    model_monthly = df_model.groupby('MonthPeriod').agg({
                        'Gen_Energy_MWh': 'sum',
                        'Settlement_$': 'sum',
                        'Market_Revenue_$': 'sum',
                        'VPPA_Payment_$': 'sum'
                    }).reset_index()
                    model_monthly['Month'] = model_monthly['MonthPeriod'].dt.strftime('%b %Y')
                    
                    # Aggregate User Bill to Monthly (handling if it's already monthly or interval)
                    user_monthly = df_bill.groupby('Month').agg({
                        'User_Gen_MW': 'sum', # If interval, this needs freq adjustment!
                        'User_Settlement_Amount': 'sum'
                    }).reset_index()
                    
                    # Fix User Generation Unit if it was Interval MW -> needs MWh
                    # If the bill was monthly summary, User_Gen_MW is likely MWh already.
                    # If it was 15-min interval MW, we need to divide by 4.
                    # Heuristic: Check number of rows.
                    if len(df_bill) > 100: # detailed interval data
                        # Estimate frequency
                        time_diff = df_bill['Time'].diff().median()
                        freq_hours = time_diff.total_seconds() / 3600.0 if pd.notnull(time_diff) else 0.25
                        user_monthly['User_Gen_MW'] = user_monthly['User_Gen_MW'] * freq_hours
                        
                    # Rename for clarity
                    user_monthly = user_monthly.rename(columns={'User_Gen_MW': 'Actual_Gen_MWh', 'User_Settlement_Amount': 'Actual_Settlement_$'})
                    
                    # Merge Comparison
                    df_comparison = pd.merge(model_monthly, user_monthly, on='Month', how='outer').fillna(0)
                    
                    # Calculate Variances
                    df_comparison['Gen_Diff_MWh'] = df_comparison['Gen_Energy_MWh'] - df_comparison['Actual_Gen_MWh']
                    df_comparison['Settlement_Diff_$'] = df_comparison['Settlement_$'] - df_comparison['Actual_Settlement_$']
                    
                    # Formatting Dashboard
                    st.markdown("### 🎯 Model Accuracy Dashboard")
                    
                    # Totals
                    tot_mod_gen = df_comparison['Gen_Energy_MWh'].sum()
                    tot_act_gen = df_comparison['Actual_Gen_MWh'].sum()
                    tot_mod_set = df_comparison['Settlement_$'].sum()
                    tot_act_set = df_comparison['Actual_Settlement_$'].sum()
                    
                    var_gen_pct = ((tot_mod_gen - tot_act_gen) / tot_act_gen) * 100 if tot_act_gen != 0 else 0
                    var_set_pct = ((tot_mod_set - tot_act_set) / tot_act_set) * 100 if tot_act_set != 0 else 0
                    
                    # Display Variances
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Modeled Generation", f"{tot_mod_gen:,.0f} MWh")
                    m2.metric("Actual Generation", f"{tot_act_gen:,.0f} MWh", delta=f"{var_gen_pct:+.1f}% Variance", delta_color="inverse")
                    
                    m3.metric("Modeled Settlement", f"${tot_mod_set:,.0f}")
                    m4.metric("Actual Settlement", f"${tot_act_set:,.0f}", delta=f"{var_set_pct:+.1f}% Variance", delta_color="inverse")
                    
                    # Charts
                    st.markdown("#### Monthly Comparison")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='Modeled Limit', x=df_comparison['Month'], y=df_comparison['Settlement_$'], marker_color='lightblue'))
                    fig.add_trace(go.Bar(name='Actual Bill', x=df_comparison['Month'], y=df_comparison['Actual_Settlement_$'], marker_color='coral'))
                    fig.update_layout(barmode='group', title='Monthly Settlement: Modeled vs Actual')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Detailed Comparison Table"):
                        st.dataframe(df_comparison.style.format({
                            'Gen_Energy_MWh': '{:,.0f}',
                            'Actual_Gen_MWh': '{:,.0f}',
                            'Settlement_$': '${:,.2f}',
                            'Actual_Settlement_$': '${:,.2f}',
                            'Market_Revenue_$': '${:,.2f}',
                            'VPPA_Payment_$': '${:,.2f}',
                            'Gen_Diff_MWh': '{:+,.0f}',
                            'Settlement_Diff_$': '${:+,.2f}'
                        }))
                        
                elif is_monthly_summary:
                    # Fallback to existing fetch & calculate logic
                    st.warning("⚠️ No recent model run found. Calculating expected settlement from scratch based on current settings.")
                    st.info("📊 Detected monthly summary data. Calculating expected totals from interval-level market data...")
                    
                    # 3. Fetch market data already filtered to the selected hub
                    df_market_hub = load_market_hub_data(
                        val_year,
                        st.session_state.get('val_hub', 'HB_NORTH')
                    )

                    if df_market_hub.empty:
                        st.error(f"Could not find market data for {val_year}.")
                    else:
                        if selected_month_numbers:
                            df_market_hub = df_market_hub[df_market_hub['Time_Central'].dt.month.isin(selected_month_numbers)]

                        # Aggregate market SPP once per month (vectorized) instead of row-wise month slicing.
                        df_market_hub['MonthPeriod'] = df_market_hub['Time_Central'].dt.to_period('M')
                        market_monthly = (
                            df_market_hub.groupby('MonthPeriod', as_index=False)['SPP']
                            .mean()
                            .rename(columns={'SPP': 'Avg_Market_Price'})
                        )

                        bill_monthly = (
                            df_bill.groupby('MonthPeriod', as_index=False)
                            .agg({
                                'User_Gen_MW': 'sum',
                                'User_Settlement_Amount': 'sum'
                            })
                        )

                        df_comparison = pd.merge(bill_monthly, market_monthly, on='MonthPeriod', how='left')
                        df_comparison = df_comparison.dropna(subset=['Avg_Market_Price']).copy()

                        if df_comparison.empty:
                            st.error("No market price data matched the uploaded bill months.")
                            st.stop()

                        revenue_share_pct = val_revenue_share / 100.0
                        if revenue_share_pct < 1.0:
                            upside = np.maximum(df_comparison['Avg_Market_Price'] - val_vppa_price, 0)
                            downside = np.minimum(df_comparison['Avg_Market_Price'] - val_vppa_price, 0)
                            settlement_per_mwh = (upside * revenue_share_pct) + downside
                        else:
                            settlement_per_mwh = df_comparison['Avg_Market_Price'] - val_vppa_price

                        df_comparison['Calculated Settlement'] = df_comparison['User_Gen_MW'] * settlement_per_mwh
                        df_comparison['Difference'] = df_comparison['Calculated Settlement'] - df_comparison['User_Settlement_Amount']
                        df_comparison['Month'] = df_comparison['MonthPeriod'].dt.strftime('%b %Y')
                        df_comparison['Generation (MWh)'] = df_comparison['User_Gen_MW']
                        df_comparison['Avg Market Price'] = df_comparison['Avg_Market_Price'].map(lambda x: f"${x:.2f}")
                        df_comparison['Strike Price'] = f"${val_vppa_price:.2f}"
                        df_comparison['User Reported'] = df_comparison['User_Settlement_Amount']

                        df_comparison = df_comparison[[
                            'Month',
                            'Generation (MWh)',
                            'Avg Market Price',
                            'Strike Price',
                            'Calculated Settlement',
                            'User Reported',
                            'Difference',
                        ]].copy()

                        st.subheader("Monthly Breakdown")
                        
                        # Display summary
                        total_calc = df_comparison['Calculated Settlement'].sum()
                        total_user = df_comparison['User Reported'].sum()
                        total_diff = total_calc - total_user
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Calculated Total", f"${total_calc:,.2f}")
                        col2.metric("User Reported Total", f"${total_user:,.2f}")
                        col3.metric("Difference", f"${total_diff:,.2f}", delta=f"{(total_diff/total_user*100):.1f}%" if total_user != 0 else "N/A")
                        
                        # Show breakdown
                        st.dataframe(df_comparison, use_container_width=True)
                        
                        # Chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=df_comparison['Month'],
                            y=df_comparison['Calculated Settlement'],
                            name='Calculated',
                            marker_color='lightblue'
                        ))
                        fig.add_trace(go.Bar(
                            x=df_comparison['Month'],
                            y=df_comparison['User Reported'],
                            name='User Reported',
                            marker_color='coral'
                        ))
                        fig.update_layout(
                            title="Monthly Settlement Comparison",
                            barmode='group',
                            yaxis_title="Settlement Amount ($)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Explanation
                        with st.expander("📘 How We Calculated"):
                            if val_revenue_share < 100:
                                st.markdown(f"""
                                **Calculation Method:**
                                1. For each month, we used your reported generation (MWh)
                                2. We calculated the average market price at **{st.session_state.get('val_hub', 'HB_NORTH')}** for that month using ERCOT RTM data
                                3. **Revenue Share Applied:** {val_revenue_share}% of upside (when SPP > ${val_vppa_price:.2f})
                                   - Upside: (Avg SPP - Strike) × {val_revenue_share}% × Generation
                                   - Downside: Full exposure (100%) when SPP < Strike
                                
                                **Formula:**
                                ```
                                When Avg SPP > Strike: Settlement = Gen × (Avg SPP - ${val_vppa_price:.2f}) × {val_revenue_share/100:.0%}
                                When Avg SPP ≤ Strike: Settlement = Gen × (Avg SPP - ${val_vppa_price:.2f})
                                ```
                                
                                **Note:** This uses monthly average pricing. For precise validation, interval-level data is recommended.
                                """)
                            else:
                                st.markdown(f"""
                                **Calculation Method:**
                                1. For each month, we used your reported generation (MWh)
                                2. We calculated the average market price at **{st.session_state.get('val_hub', 'HB_NORTH')}** for that month using ERCOT RTM data
                                3. Settlement = Generation × (Avg Market Price - Strike Price)
                                
                                **Formula:**
                                ```
                                Monthly Settlement = Generation (MWh) × (Avg SPP - ${val_vppa_price:.2f})
                                ```
                                
                                **Note:** This uses monthly average pricing. For precise validation, interval-level data is recommended.
                                """)
                
                else:
                    # Original interval-level validation logic
                    # 3. Fetch market data already filtered to the selected hub
                    df_market_hub = load_market_hub_data(
                        val_year,
                        st.session_state.get('val_hub', 'HB_NORTH')
                    )

                    if df_market_hub.empty:
                        st.error(f"Could not find market data for {val_year}.")
                    else:
                        if selected_month_numbers:
                            df_market_hub = df_market_hub[df_market_hub['Time_Central'].dt.month.isin(selected_month_numbers)]

                        # Narrow the market slice to uploaded timestamps before merging.
                        min_user_time = df_bill['Time'].min()
                        max_user_time = df_bill['Time'].max()
                        df_market_hub = df_market_hub[
                            (df_market_hub['Time'] >= min_user_time) &
                            (df_market_hub['Time'] <= max_user_time)
                        ]
                        
                        # Merge on Time (tolerance?)
                        # Let's use merge_asof if indices are sorted, or simple merge
                        df_bill = df_bill.sort_values('Time')
                        df_market_hub = df_market_hub.sort_values('Time')
                        
                        # Using merge directly (exact match). If 15-min intervals match.
                        df_merged = pd.merge(df_bill, df_market_hub[['Time', 'SPP', 'Time_Central']], on='Time', how='inner')
                        
                        if df_merged.empty:
                            st.error("❌ No matching timestamps found between User Data and Market Data. Check your year and timestamp format.")
                        else:
                            st.success(f"✅ Successfully matched {len(df_merged):,} intervals.")
                            
                            # 5. Calculate Expected Settlement
                            # Interval hours (assuming 15-min data if mostly consecutive)
                            # Detect frequency?
                            time_diff = df_merged['Time'].diff().median()
                            freq_hours = time_diff.total_seconds() / 3600.0 if pd.notnull(time_diff) else 0.25 # Default 15 min
                            
                            # Assume data is POWER (MW) -> Energy (MWh) = MW * hours
                            # If user data is already MWh? Column name might hint, but usually profiles are MW.
                            # Let's assume MW.
                            
                            df_merged['Calculated_Gen_MWh'] = df_merged['User_Gen_MW'] * freq_hours
                            df_merged['Strike_Price'] = val_vppa_price
                            
                            # Apply revenue share logic
                            revenue_share_pct = val_revenue_share / 100.0
                            if revenue_share_pct < 1.0:
                                # When SPP > VPPA: Settlement = (SPP - VPPA) * share_pct (buyer gets only their share of upside)
                                # When SPP <= VPPA: Settlement = SPP - VPPA (full downside, no sharing)
                                upside = np.maximum(df_merged['SPP'] - val_vppa_price, 0)
                                downside = np.minimum(df_merged['SPP'] - val_vppa_price, 0)
                                settlement_price = (upside * revenue_share_pct) + downside
                            else:
                                settlement_price = df_merged['SPP'] - val_vppa_price
                            
                            df_merged['Market_Revenue'] = df_merged['Calculated_Gen_MWh'] * df_merged['SPP']
                            df_merged['Fixed_Revenue'] = df_merged['Calculated_Gen_MWh'] * val_vppa_price
                            df_merged['Expected_Settlement'] = df_merged['Calculated_Gen_MWh'] * settlement_price
                            
                            # 6. Display Results
                            
                            # Metrics
                            total_gen_mwh = df_merged['Calculated_Gen_MWh'].sum()
                            total_settlement = df_merged['Expected_Settlement'].sum()
                            avg_spp = df_merged['SPP'].mean()
                            
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Total Generation", f"{total_gen_mwh:,.0f} MWh")
                            m2.metric("Calculated Net Settlement", f"${total_settlement:,.2f}")
                            m3.metric("Avg Hub Price (SPP)", f"${avg_spp:.2f}/MWh")
                            
                            # Comparison if User provided settlement
                            if 'User_Settlement_Amount' in df_merged.columns:
                                user_total = df_merged['User_Settlement_Amount'].sum()
                                diff = total_settlement - user_total
                                st.write(f"**Discrepancy:** ${diff:,.2f} (Calculated - User Uploaded)")
                                if abs(diff) > 100:
                                    st.warning("Significant discrepancy detected.")
                                else:
                                    st.success("Matches closely!")

                            # 7. Visualization
                            st.subheader("Settlement Over Time")
                            
                            # Aggregate to Daily for cleaner chart
                            df_merged['Date'] = df_merged['Time_Central'].dt.date
                            daily_df = df_merged.groupby('Date')[['Expected_Settlement', 'Market_Revenue']].sum().reset_index()
                            
                            fig = px.bar(daily_df, x='Date', y='Expected_Settlement', title="Daily Net Settlement")
                            fig.add_hline(y=0, line_dash="dash", line_color="black")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            with st.expander("View Detailed Data"):
                                st.dataframe(df_merged[['Time_Central', 'User_Gen_MW', 'SPP', 'Expected_Settlement']])
                                
                            # Download Results
                            csv = df_merged.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Validation Results CSV",
                                data=csv,
                                file_name="bill_validation_results.csv",
                                mime="text/csv",
                            )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            import traceback
            st.code(traceback.format_exc())




with tab_performance:
    st.header("🎯 Model Performance & Benchmarking")
    st.markdown("""
    This tab showcases the accuracy of our **high-fidelity synthetic generation models**. 
    We benchmark our profiles against actual **ERCOT SCED (Security Constrained Economic Dispatch)** generation data for 2024.
    """)

    with st.expander("📖 Understanding the Metrics", expanded=False):
        st.markdown("""
        **1. Correlation Coefficient (R)**
        - **What it means:** Measures how well the *shape* of our model matches reality.
        - **Range:** -1.0 to 1.0. (1.0 is a perfect match).
        - **Good Score:** > 0.90 is excellent.
        
        **2. Coefficient of Determination (R^2)**
        - **What it means:** The percentage of the variability in actual generation that our model explains.
        - **Analogy:** If R^2 = 0.95, our model captures 95% of the ups and downs correctly.
        
        **3. Mean Bias Error (MBE)**
        - **What it means:** The average difference between Model and Actual.
        - **Positive:** Model thinks it's sunnier/windier than it is (Over-prediction).
        - **Negative:** Model is too conservative (Under-prediction).
        
        **4. Root Mean Square Error (RMSE)**
        - **What it means:** The "standard deviation" of the error. Penalizes big misses more than small ones.
        - **Use:** Lower is better. Tells you the typical error margin in MW.
        """)

    # Load Results
    try:
        with open('benchmark_results_wind.json', 'r') as f:
            wind_res = pd.DataFrame(json.load(f))
        with open('benchmark_results_solar.json', 'r') as f:
            solar_res = pd.DataFrame(json.load(f))
    except Exception as e:
        st.error(f"Error loading benchmark results: {e}")
        st.stop()

    coll1, coll2 = st.tabs(["💨 Wind Performance", "☀️ Solar Performance"])

    with coll1:
        st.subheader("Wind Model Benchmarking (Dec '24 - Nov '25)")
        
        # Metrics Overview
        # Filter for Advanced model
        wind_advanced = wind_res[wind_res['Model'].str.contains('Advanced')]
        avg_r_wind = wind_advanced['R'].mean()
        max_r_wind = wind_advanced['R'].max()
        
        m1, m2 = st.columns(2)
        m1.metric(
            "Avg Correlation (R)",
            f"{avg_r_wind:.2f}" if pd.notna(avg_r_wind) else "N/A",
            help="Correlation between synthetic and actual generation"
        )
        m2.metric("Top Correlation", f"{max_r_wind:.2f}" if pd.notna(max_r_wind) else "N/A")

        st.markdown("### 🏆 Wind Leaderboard (Advanced Model)")
        top_wind = wind_advanced.sort_values('R', ascending=False).head(10)
        
        # Prepare columns for display, ensuring new metrics exist
        display_cols = ['Project', 'R', 'MBE (MW)', 'RMSE (MW)']
        extra_cols = ['R_Hourly', 'R_Daily']
        for c in extra_cols:
            if c in top_wind.columns:
                display_cols.insert(2, c) # Insert after R
                
        st.dataframe(
            top_wind[display_cols],
            column_config={
                "R": st.column_config.NumberColumn(
                    "R (15m)",
                    help="Pearson Correlation (15-min)",
                    format="%.2f"
                ),
                "R_Hourly": st.column_config.NumberColumn(
                    "R (Hr)",
                    help="Pearson Correlation (Hourly)",
                    format="%.2f"
                ),
                "R_Daily": st.column_config.NumberColumn(
                    "R (Day)",
                    help="Pearson Correlation (Daily)",
                    format="%.2f"
                ),
                "MBE (MW)": st.column_config.NumberColumn(
                    "MBE (MW)",
                    help="Mean Bias Error. The average difference between Modeled and Actual MW. Positive = Model Overestimates, Negative = Model Underestimates.",
                    format="%.2f"
                ),
                "RMSE (MW)": st.column_config.NumberColumn(
                    "RMSE (MW)",
                    help="Root Mean Square Error. Measures the typical magnitude of error in MW, penalizing larger errors more heavily.",
                    format="%.2f"
                )
            },
            use_container_width=True
        )

        st.info("💡 **Insight:** Advanced models (using actual hub heights and turbine curves) reduce bias by ~15% on average compared to baseline models.")

    with coll2:
        st.subheader("Solar Model Benchmarking (Dec '24 - Nov '25)")
        
        # Metrics Overview
        solar_advanced = solar_res[solar_res['Model'].str.contains('Advanced')]
        avg_r_solar = solar_advanced['R'].mean()
        max_r_solar = solar_advanced['R'].max()
        
        m1, m2 = st.columns(2)
        m1.metric("Avg Correlation (R)", f"{avg_r_solar:.2f}" if pd.notna(avg_r_solar) else "N/A")
        m2.metric("Top Correlation", f"{max_r_solar:.2f}" if pd.notna(max_r_solar) else "N/A")

        st.markdown("### 🏆 Solar Leaderboard (Tracking Model)")
        top_solar = solar_advanced.sort_values('R', ascending=False).head(10)
        # Prepare columns for display, ensuring new metrics exist
        display_cols = ['Project', 'R', 'MBE (MW)', 'RMSE (MW)']
        extra_cols = ['R_Hourly', 'R_Daily']
        for c in extra_cols:
            if c in top_solar.columns:
                display_cols.insert(2, c) # Insert after R

        st.dataframe(
            top_solar[display_cols],
            column_config={
                "R": st.column_config.NumberColumn(
                    "R (15m)",
                    help="Pearson Correlation (15-min)",
                    format="%.2f"
                ),
                "R_Hourly": st.column_config.NumberColumn(
                    "R (Hr)",
                    help="Pearson Correlation (Hourly)",
                    format="%.2f"
                ),
                "R_Daily": st.column_config.NumberColumn(
                    "R (Day)",
                    help="Pearson Correlation (Daily)",
                    format="%.2f"
                ),
                "MBE (MW)": st.column_config.NumberColumn(
                    "MBE (MW)",
                    help="Mean Bias Error. The average difference between Modeled and Actual MW. Positive = Model Overestimates, Negative = Model Underestimates.",
                    format="%.2f"
                ),
                "RMSE (MW)": st.column_config.NumberColumn(
                    "RMSE (MW)",
                    help="Root Mean Square Error. Measures the typical magnitude of error in MW, penalizing larger errors more heavily.",
                    format="%.2f"
                )
            },
            use_container_width=True
        )

        st.success("✅ **Key Finding:** Solar generation is highly predictable (R > 0.85) when accounting for single-axis tracking gains.")




    st.divider()
    st.subheader("🔍 Project Deep Dive & Benchmarking")
    st.markdown("Select an ERCOT project to retrieve actual SCED generation data and compare it against our high-fidelity synthetic model.")

    # Load registry and candidate list
    try:
        with open('ercot_assets.json', 'r') as f:
            asset_registry = json.load(f)
    except Exception:
        asset_registry = {}
        
    try:
        with open('ercot_renewable_assets.txt', 'r') as f:
            candidate_units = [line.strip() for line in f if line.strip()]
    except Exception:
        candidate_units = []

    # Asset Selection Logic
    col_reg, col_cust = st.columns([0.6, 0.4])
    with col_reg:
        curated_options = ["None (Use Custom ID)"] + list(asset_registry.keys())
        default_curated_index = 1 if len(curated_options) > 1 else 0
        benchmark_asset_name = st.selectbox(
            "Select Curated Project",
            options=curated_options,
            index=default_curated_index
        )
    with col_cust:
        custom_resource_id = st.text_input("OR Enter Custom Resource ID (from ercot_renewable_assets.txt)", help="Example: FRYE_SLR_UNIT1, MONTECR1_WIND1")

    # Determine target asset
    final_resource_id = None
    asset_meta = None
    
    if custom_resource_id:
        final_resource_id = custom_resource_id.strip().upper()
        # Try to find tech/lat/lon if it happens to be in our registry
        for name, meta in asset_registry.items():
            if meta['resource_name'] == final_resource_id:
                asset_meta = meta
                break
    elif benchmark_asset_name != "None (Use Custom ID)":
        asset_meta = asset_registry[benchmark_asset_name]
        final_resource_id = asset_meta['resource_name']
        
        # --- BENCHMARK DATA INTEGRATION ---
        # "Load project data from model performance"
        # We try to find this project in our benchmark results JSONs
        found_bench = []
        try:
            with open('benchmark_results_wind.json', 'r') as f:
                wand = json.load(f)
                found_bench.extend([r for r in wand if r.get('Project') == benchmark_asset_name])
            with open('benchmark_results_solar.json', 'r') as f:
                sand = json.load(f)
                found_bench.extend([r for r in sand if r.get('Project') == benchmark_asset_name])
        except Exception:
            pass
            
        if found_bench:
            # Sort by R descending to find "Best"
            def bench_r_or_floor(row):
                val = row.get('R')
                return float(val) if pd.notna(val) else -1.0
            best_run = sorted(found_bench, key=bench_r_or_floor, reverse=True)[0]
            
            with st.expander(f"🏆 Benchmark Performance ({best_run.get('Model')})", expanded=True):
                # Row 1: Key Metrics
                c_b1, c_b2, c_b3 = st.columns(3)
                r_val = best_run.get('R')
                mbe_val = best_run.get('MBE (MW)')
                rmse_val = best_run.get('RMSE (MW)')
                
                c_b1.metric("Correlation (15-min)", f"{r_val:.2f}" if pd.notna(r_val) else "N/A", help="Pearson R at 15-min Base Resolution")
                c_b2.metric("Mean Bias (MBE)", f"{mbe_val:.1f} MW" if pd.notna(mbe_val) else "N/A", help="Positive = Model Overpredicts")
                c_b3.metric("RMSE", f"{rmse_val:.1f} MW" if pd.notna(rmse_val) else "N/A", help="Typical Error Magnitude")
                
                # Row 2: Correlation Granularity
                st.markdown("##### 📈 Correlation by Time Scale")
                c_r1, c_r2, c_r3 = st.columns(3)
                
                r_hour = best_run.get('R_Hourly')
                r_day = best_run.get('R_Daily')
                
                c_r1.metric("Hourly R", f"{r_hour:.2f}" if pd.notna(r_hour) else "N/A", help="Correlation of Hourly Averages")
                c_r2.metric("Daily R", f"{r_day:.2f}" if pd.notna(r_day) else "N/A", help="Correlation of Daily Totals")
                c_r3.caption("Aggregating to Daily usually improves correlation by smoothing out short-term timing mismatches.")
                
                st.caption(f"**Best Configuration:** {best_run.get('Model')}")
                
                # Download Options
                # Combine Wind and Solar loaded data for full download
                all_bench_data = []
                try:
                    with open('benchmark_results_wind.json', 'r') as f:
                        all_bench_data.extend(json.load(f))
                    with open('benchmark_results_solar.json', 'r') as f:
                        all_bench_data.extend(json.load(f))
                except:
                    pass
                
                if all_bench_data:
                    df_bench_all = pd.DataFrame(all_bench_data)
                    csv_bench = df_bench_all.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Benchmark Study (All Assets)",
                        data=csv_bench,
                        file_name="ercot_renewable_benchmark_results.csv",
                        mime="text/csv",
                        help="Download the complete performance dataset for all Wind and Solar assets."
                    )

    # Data Source Selection
    val_source = st.radio("Validation Data Source", ["ERCOT Public SCED (60-day delay)", "Upload Private Data (CSV/Excel)"], horizontal=True, index=0)
    
    uploaded_val_file = None
    if val_source == "Upload Private Data (CSV/Excel)":
        uploaded_val_file = st.file_uploader("Upload Generation Data", type=["csv", "xlsx"], help="Columns required: 'Time' (datetime) and 'Actual_MW' (or 'Gen_MW').")
        if uploaded_val_file:
             st.info("File uploaded. Click 'Fetch & Benchmark' to process.")

    if final_resource_id:
        # Date Range Picker
        max_date = (datetime.now() - timedelta(days=65)).date()
        min_date = max_date - timedelta(days=365) # Allow 1 year lookback within disclosure
        
        st.markdown(f"**Targeting:** `{final_resource_id}`")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_bench = st.date_input("Start Date", value=max_date - timedelta(days=1), min_value=min_date, max_value=max_date)
        with col_d2:
            end_bench = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

        if asset_meta:
            # Rich Metadata Display
            with st.container():
                st.markdown(f"#### 📍 {asset_meta.get('project_name', final_resource_id)}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Capacity", f"{asset_meta['capacity_mw']} MW")
                m2.metric("Technology", asset_meta['tech'])
                m3.metric("Hub", asset_meta['hub'])
                m4.metric("County", asset_meta.get('county', 'Unknown'))
                
                # Coordinates with small map link conceptual
                st.caption(f"**Coordinates:** {asset_meta['lat']:.4f}, {asset_meta['lon']:.4f}")
                if 'turbine_model' in asset_meta:
                     st.caption(f"**Turbine:** {asset_meta.get('turbine_manuf', '')} {asset_meta['turbine_model']} (found via USWTDB)")
                     
        else:
            st.warning("⚠️ Manual Override: Metadata (lat/lon) missing for this ID. Modeling comparison will use current scenario settings.")

        
        # --- Advanced Model Parameters ---
        bench_wind_weather_source = "AUTO"
        bench_hrrr_forecast_hour = 0
        bench_wind_model_engine = "STANDARD"
        with st.expander("🛠️ Advanced Model Parameters"):
            c_p1, c_p2, c_p3 = st.columns(3)
            # Losses
            model_losses = c_p1.slider("System Losses (%)", 0, 50, 14, help="Reduces gross modeled output (Wake, Electrical, Availability). Default ~14%.")
            
            # Bias Correction
            model_bias = c_p2.number_input("Linear Bias Correction (Multiplier)", 0.5, 1.5, 1.0, step=0.01, help="Scalar to linearly tune model up/down.")
            
            # Turbine Type
            turbine_opts = ["Auto-Detect", "Generic (IEC Class 2)", "Vestas V163 (Low Wind)", "GE 2.x (Workhorse)", "GE 3.6-154 (Modern Mainstream)", "Nordex N163 (5.X MW)", "Siemens-Gamesa SG 3.4-132"]
            selected_turb = c_p3.selectbox("Turbine Type Override", turbine_opts)
            
            turbine_override_map = {
                "Auto-Detect": None,
                "Generic (IEC Class 2)": "GENERIC",
                "Vestas V163 (Low Wind)": "VESTAS_V163",
                "GE 2.x (Workhorse)": "GE_2X",
                "GE 3.6-154 (Modern Mainstream)": "GE_3X",
                "Nordex N163 (5.X MW)": "NORDEX_N163",
                "Siemens-Gamesa SG 3.4-132": "SG_3_4_132",
            }
            final_turbine_req = turbine_override_map[selected_turb]

            c_w1, c_w2, c_w3 = st.columns([2, 1, 2])
            with c_w1:
                bench_wind_label = st.selectbox(
                    "Wind Weather Dataset",
                    list(WIND_WEATHER_SOURCE_OPTIONS.keys()),
                    key="bench_wind_weather_source_label",
                    help="NOAA HRRR uses cached files under data_cache/hrrr.",
                )
                bench_wind_weather_source = WIND_WEATHER_SOURCE_OPTIONS.get(bench_wind_label, "AUTO")
            with c_w2:
                if bench_wind_weather_source == "NOAA_HRRR_CACHED":
                    bench_hrrr_forecast_hour = int(
                        st.number_input(
                            "HRRR fxx",
                            min_value=0,
                            max_value=18,
                            value=0,
                            step=1,
                            key="bench_hrrr_forecast_hour",
                        )
                    )
                    st.caption("0=analysis")
                    hrrr_count = get_hrrr_cache_count()
                    if hrrr_count == 0:
                        st.warning("No HRRR cache files found yet. Generate them with `scripts/fetch_hrrr_wind.py`.")
                    else:
                        st.caption(f"Detected {hrrr_count} cached HRRR files.")
            with c_w3:
                bench_engine_label = st.selectbox(
                    "Wind Model Engine",
                    list(WIND_MODEL_ENGINE_OPTIONS.keys()),
                    index=0,
                    key="bench_wind_model_engine_label",
                    help="Advanced mode applies monthly EIA/CF targets, SCED bias correction, node adjustments, and tuned clipping.",
                )
                bench_wind_model_engine = WIND_MODEL_ENGINE_OPTIONS.get(bench_engine_label, "STANDARD")

        if st.button(f"🚀 Fetch & Benchmark {final_resource_id}"):
            if start_bench > end_bench:
                st.error("Start date must be before end date.")
            else:
                with st.spinner(f"Retrieving actual production and running model for {final_resource_id}..."):
                    # 1. Fetch Actual Data
                    if val_source == "Upload Private Data (CSV/Excel)" and uploaded_val_file:
                        try:
                            if uploaded_val_file.name.endswith('.csv'):
                                df_actual = pd.read_csv(uploaded_val_file)
                            else:
                                df_actual = pd.read_excel(uploaded_val_file)
                            
                            # Standardize Columns
                            # Look for time col
                            time_col = next((c for c in df_actual.columns if 'time' in c.lower() or 'date' in c.lower()), None)
                            gen_col = next((c for c in df_actual.columns if 'mw' in c.lower() or 'gen' in c.lower() or 'actual' in c.lower()), None)
                            
                            if not time_col or not gen_col:
                                st.error(f"Could not identify Time/Gen columns. Found: {df_actual.columns.tolist()}")
                                df_actual = pd.DataFrame()
                            else:
                                df_actual = df_actual.rename(columns={time_col: 'Time', gen_col: 'Actual_MW'})
                                df_actual['Time'] = pd.to_datetime(df_actual['Time'], errors='coerce')
                                df_actual['Actual_MW'] = pd.to_numeric(df_actual['Actual_MW'], errors='coerce')
                                # Normalize to UTC so downstream merge uses consistent timezone-aware datetimes.
                                if df_actual['Time'].dt.tz is None:
                                    df_actual['Time'] = (
                                        df_actual['Time']
                                        .dt.tz_localize('US/Central', ambiguous='NaT', nonexistent='shift_forward')
                                        .dt.tz_convert('UTC')
                                    )
                                else:
                                    df_actual['Time'] = df_actual['Time'].dt.tz_convert('UTC')
                                df_actual = df_actual.dropna(subset=['Time', 'Actual_MW']).copy()
                                
                        except Exception as e:
                            st.error(f"Error parsing file: {e}")
                            df_actual = pd.DataFrame()
                    else:
                        # Public SCED
                        df_actual = get_cached_asset_data_with_base_point(final_resource_id, start_bench, end_bench)
                    
                    if not df_actual.empty:
                        # 2. Run Model for comparison
                        # Use metadata if available, otherwise use page settings (Analysis Configuration)
                        compare_lat = asset_meta['lat'] if asset_meta else st.session_state.get('val_custom_lat', 32.0)
                        compare_lon = asset_meta['lon'] if asset_meta else st.session_state.get('val_custom_lon', -100.0)
                        
                        # Use Preview settings if meta missing
                        compare_tech = asset_meta['tech'] if asset_meta else st.session_state.get('preview_tech', 'Solar')
                        compare_cap = asset_meta['capacity_mw'] if asset_meta else st.session_state.get('preview_capacity', 100.0)
                        
                        # Use enriched model first, then manual type, then generic
                        # If override is set, use it.
                        if final_turbine_req:
                            compare_turbine = final_turbine_req
                        elif asset_meta:
                            compare_turbine = asset_meta.get('turbine_model', asset_meta.get('turbine_type', 'GENERIC'))
                        else:
                            # Use the preview selector if no asset meta
                            compare_turbine = selected_turbine if selected_turbine != "GENERIC" else "GENERIC"
                        
                        # Normalize actual timestamps to UTC for safe alignment/merge.
                        if df_actual['Time'].dt.tz is None:
                            df_actual['Time'] = df_actual['Time'].dt.tz_localize('UTC')
                        else:
                            df_actual['Time'] = df_actual['Time'].dt.tz_convert('UTC')

                        # Run model for all years in range
                        target_years = sorted(list(set([d.year for d in pd.to_datetime(df_actual['Time'])])))
                        model_dfs = []
                        for yr in target_years:
                            if compare_tech == "Wind":
                                m_df = fetch_tmy.get_profile_for_year(
                                    yr,
                                    compare_tech,
                                    compare_cap,
                                    compare_lat,
                                    compare_lon,
                                    turbine_type=compare_turbine,
                                    efficiency=1.0,
                                    hub_name=asset_meta.get('hub') if asset_meta else None,
                                    project_name=asset_meta.get('project_name') if asset_meta else None,
                                    resource_id=final_resource_id,
                                    apply_wind_calibration=True,
                                    turbines=asset_meta.get('turbines') if asset_meta else None,
                                    wind_weather_source=bench_wind_weather_source,
                                    hrrr_forecast_hour=bench_hrrr_forecast_hour,
                                    wind_model_engine=bench_wind_model_engine,
                                )
                            else:
                                m_df = fetch_tmy.get_profile_for_year(
                                    yr,
                                    compare_tech,
                                    compare_cap,
                                    compare_lat,
                                    compare_lon,
                                    turbine_type=compare_turbine,
                                    efficiency=1.0,
                                )
                            model_dfs.append(m_df)
                        df_modeled_full = pd.concat(model_dfs).to_frame(name='Gen_MW')
                        
                        # Slice to match
                        actual_times = df_actual['Time']
                        # Ensure timezone alignment before slicing/merging
                        if df_modeled_full.index.tz is None:
                            df_modeled_full.index = df_modeled_full.index.tz_localize('UTC')
                        if actual_times.dt.tz is None:
                            actual_times = actual_times.dt.tz_localize('UTC')
                            
                        # Normalize to same TZ (UTC) just in case
                        df_modeled_full.index = df_modeled_full.index.tz_convert('UTC')
                        
                        # Apply Custom Scalars (Losses & Bias)
                        # Gen_MW is gross, apply losses: Gen_Net = Gen_Gross * (1 - losses) * bias
                        efficiency_factor = (1 - model_losses / 100.0)
                        df_modeled_full['Gen_MW'] = df_modeled_full['Gen_MW'] * efficiency_factor * model_bias
                        
                        df_modeled_slice = df_modeled_full[df_modeled_full.index.isin(actual_times)].copy()
                        
                        # 3. Merge and Compare
                        df_comp = pd.merge(df_actual, df_modeled_slice.reset_index().rename(columns={'index': 'Time'}), on='Time')
                        df_comp = df_comp.rename(columns={'Gen_MW': 'Modeled_MW'})
                        
                        # Drop any potential NaNs from resampling gaps to avoid Metric errors
                        df_comp = df_comp.dropna(subset=['Actual_MW', 'Modeled_MW'])
                        
                        # Save to Session State
                        st.session_state['bench_results'] = {
                            'df_comp': df_comp,
                            'df_actual': df_actual,
                            'resource_id': final_resource_id,
                            'start': start_bench,
                            'end': end_bench,
                            'turbine': compare_turbine,
                            'tech': compare_tech,
                            'capacity_mw': compare_cap,
                        }
                    else:
                        st.error(f"No generation data found for `{final_resource_id}` in this period. Note: Data stops ~60 days before today.")
        
        # --- Visualization Section (Outside Button Logic) ---
        if 'bench_results' in st.session_state:
            res = st.session_state['bench_results']
            
            st.divider()
            st.markdown(f"### 📊 Results for `{res['resource_id']}`")
            
            df_comp = res['df_comp'].copy()
            
            # Time Granularity Selector
            st.markdown("#### ⏱️ Time Resolution")
            
            # Economic Dispatch Toggle
            # col_t1, col_t2 = st.columns([2, 1])
            # with col_t1:
            granularity = st.radio("Select View:", ["15-Minute", "Hourly", "Daily", "Monthly", "Annual"], horizontal=True, index=0)
            # with col_t2:
            apply_dispatch = st.checkbox(
                "📉 Apply Economic Dispatch",
                value=True,
                help="Limit Modeled Generation to Base Point (Grid Limit) to simulate curtailment. Enabled by default to match leaderboard methodology.",
                key=f"chk_disp_{res['resource_id']}"
            )
            dispatch_headroom_factor = 1.0
            dispatch_cap_strength = DEFAULT_BASE_POINT_CAP_STRENGTH
            if 'Base_Point_MW' in df_comp.columns:
                auto_headroom = estimate_base_point_headroom_factor(
                    df_comp,
                    capacity_mw=res.get('capacity_mw'),
                )
                dispatch_headroom_pct = st.slider(
                    "Base Point Headroom (%)",
                    min_value=100,
                    max_value=135,
                    value=100,
                    step=1,
                    help=(
                        "Effective cap = Base Point × headroom. "
                        "Default is 100%; auto suggestion from historical Actual/Base Point is shown below."
                    ),
                    key=f"bp_headroom_{res['resource_id']}",
                )
                dispatch_headroom_factor = float(dispatch_headroom_pct) / 100.0
                st.caption(f"Auto headroom suggestion: {auto_headroom:.2f}x")
                dispatch_cap_strength_pct = st.slider(
                    "Dispatch Cap Strength (%)",
                    min_value=0,
                    max_value=100,
                    value=int(round(DEFAULT_BASE_POINT_CAP_STRENGTH * 100)),
                    step=5,
                    help=(
                        "0% leaves model uncapped. 100% is a hard Base Point cap. "
                        "Use 20-40% for a softer cap that reduces over-curtailment bias."
                    ),
                    key=f"bp_cap_strength_{res['resource_id']}",
                )
                dispatch_cap_strength = float(dispatch_cap_strength_pct) / 100.0
            exclude_offline = st.checkbox(
                "🛠️ Exclude Likely Offline Intervals",
                value=True,
                help="Exclude intervals where Actual is near zero but Model is high, using a capacity-aware threshold.",
                key=f"chk_offline_{res['resource_id']}"
            )
            
            # Apply Dispatch Logic (Curtail Model)
            if apply_dispatch and 'Base_Point_MW' in df_comp.columns:
                # If Base Point is NaN (rare), assume unconstrained (infinity), effectively keeping Modeled_MW
                # Or fill with Capacity? Let's keep it safe.
                # Actually, SCED usually has values. If NaN, it might mean data missing.
                # Logic: Modeled_Curtailed = min(Modeled, Base_Point)
                df_comp['Modeled_MW_Raw'] = df_comp['Modeled_MW'] # Keep raw for reference
                df_comp['Modeled_MW'] = apply_base_point_cap(
                    modeled_mw=df_comp['Modeled_MW'],
                    base_point_mw=df_comp['Base_Point_MW'],
                    headroom_factor=dispatch_headroom_factor,
                    capacity_mw=res.get('capacity_mw'),
                    cap_strength=dispatch_cap_strength,
                )
                bp_cov = float(pd.to_numeric(df_comp['Base_Point_MW'], errors='coerce').notna().mean()) if len(df_comp) else 0.0
                st.caption(
                    "⚠️ **Economic Dispatch Applied:** Modeled generation is capped at "
                    f"Base Point × {dispatch_headroom_factor:.2f} "
                    f"with cap strength {dispatch_cap_strength:.2f} "
                    f"(Base Point coverage {bp_cov:.1%})."
                )
            elif apply_dispatch:
                st.warning("⚠️ Base Point data missing for this period. Cannot apply economic dispatch.")

            if exclude_offline:
                offline_threshold = get_offline_threshold_mw(res.get('capacity_mw'))
                valid_mask = ~((df_comp['Actual_MW'] < 0.5) & (df_comp['Modeled_MW'] > offline_threshold))
                dropped = int((~valid_mask).sum())
                df_comp = df_comp.loc[valid_mask].copy()
                if dropped > 0:
                    st.caption(
                        f"Filtered {dropped:,} likely offline intervals "
                        f"(offline threshold {offline_threshold:.1f} MW)."
                    )

            if df_comp.empty:
                st.warning("No benchmark intervals available after filters. Adjust filters or date range.")
                st.stop()
                        
            # Prepare data for aggregation
            time_deltas = (
                df_comp['Time']
                .sort_values()
                .diff()
                .dt.total_seconds()
                .div(3600.0)
            )
            interval_hours = time_deltas[time_deltas > 0].median()
            if pd.isna(interval_hours) or interval_hours <= 0 or interval_hours > 4:
                interval_hours = 0.25
            df_comp['Actual_MWh'] = df_comp['Actual_MW'] * interval_hours
            df_comp['Modeled_MWh'] = df_comp['Modeled_MW'] * interval_hours
            
            # Resampling Logic
            if granularity == "15-Minute":
                df_agg = df_comp.copy()
                y_col_act = 'Actual_MW'
                y_col_mod = 'Modeled_MW'
                unit = "MW"
                time_col = 'Time'
            
            elif granularity == "Hourly":
                df_agg = df_comp.resample('h', on='Time').mean().reset_index()
                y_col_act = 'Actual_MW'
                y_col_mod = 'Modeled_MW'
                unit = "MW (Avg)"
                time_col = 'Time'
                
            elif granularity == "Daily":
                df_agg = df_comp.resample('D', on='Time')[['Actual_MWh', 'Modeled_MWh']].sum().reset_index()
                y_col_act = 'Actual_MWh'
                y_col_mod = 'Modeled_MWh'
                unit = "MWh (Total)"
                time_col = 'Time'
                
            elif granularity == "Monthly":
                df_agg = df_comp.resample('ME', on='Time')[['Actual_MWh', 'Modeled_MWh']].sum().reset_index()
                y_col_act = 'Actual_MWh'
                y_col_mod = 'Modeled_MWh'
                unit = "MWh (Total)"
                time_col = 'Time'
                
            elif granularity == "Annual":
                df_agg = df_comp.resample('YE', on='Time')[['Actual_MWh', 'Modeled_MWh']].sum().reset_index()
                y_col_act = 'Actual_MWh'
                y_col_mod = 'Modeled_MWh'
                unit = "MWh (Total)"
                time_col = 'Time'
            
            # Calculate Metrics on Aggregated Data
            if not df_agg.empty:
                mae = (df_agg[y_col_act] - df_agg[y_col_mod]).abs().mean()
                
                # Display Metrics row 1: Totals (Scale Independent essentially, apart from small resampling diffs)
                st.markdown("#### 📊 Numerical Summary")
                mc1, mc2, mc3 = st.columns(3)
                
                # Display Totals in MWh (always useful) or Avg MW depending on view? 
                # Let's keep totals as MWh for clarity on volume
                total_actual_mwh = df_comp['Actual_MWh'].sum()
                total_modeled_mwh = df_comp['Modeled_MWh'].sum()
                diff_pct = ((total_modeled_mwh - total_actual_mwh) / total_actual_mwh) if total_actual_mwh > 0 else np.nan
                
                mc1.metric("Total Actual Gen", f"{total_actual_mwh:,.1f} MWh")
                if pd.notna(diff_pct):
                    mc2.metric("Total Model Est", f"{total_modeled_mwh:,.1f} MWh", delta=f"{diff_pct:+.1%}", delta_color="inverse")
                else:
                    mc2.metric("Total Model Est", f"{total_modeled_mwh:,.1f} MWh", delta="N/A")
                mc3.metric("Total Bias", f"{total_modeled_mwh - total_actual_mwh:+.1f} MWh")

                # Calculate correlation metrics with explicit N/A behavior for insufficient/constant data.
                if len(df_agg) > 1 and df_agg[y_col_act].nunique() > 1 and df_agg[y_col_mod].nunique() > 1:
                    pearson_r = df_agg[y_col_act].corr(df_agg[y_col_mod])
                else:
                    pearson_r = np.nan
                r2 = pearson_r ** 2 if pd.notna(pearson_r) else np.nan
                
                # Display Metrics row 2: Statistical Fit (Context Dependent)
                m1, m2, m3 = st.columns(3)
                m1.metric(f"Mean Abs Error ({unit})", f"{mae:.1f}")
                m2.metric("Correlation (R)", f"{pearson_r:.2f}" if pd.notna(pearson_r) else "N/A", help="Pearson Correlation Coefficient")
                m3.metric("Determination (R²)", f"{r2:.2f}" if pd.notna(r2) else "N/A")
                
                # Visual Overlay
                fig_bench = go.Figure()
                
                # Determine graph type based on granularity
                if granularity in ["Daily", "Monthly", "Annual"]:
                    fig_bench.add_trace(go.Bar(x=df_agg[time_col], y=df_agg[y_col_act], name=f'Actual ({unit})', marker_color='orange', opacity=0.7))
                    # For model comparison in bars, maybe line or separate bars? Let's use Line for model to overlay neatly
                    fig_bench.add_trace(go.Scatter(x=df_agg[time_col], y=df_agg[y_col_mod], name=f'Model ({unit})', line=dict(color='blue', width=3)))
                else:
                    fig_bench.add_trace(go.Scatter(x=df_agg[time_col], y=df_agg[y_col_act], name=f'Actual ({unit})', line=dict(color='orange', width=2)))
                    fig_bench.add_trace(go.Scatter(x=df_agg[time_col], y=df_agg[y_col_mod], name=f'Model ({unit})', line=dict(color='blue', dash='dash', width=1.5)))
                
                # Grid Limit only relevant for 15-min or Hourly MW
                if 'Base_Point_MW' in df_comp.columns and granularity in ["15-Minute", "Hourly"]:
                        # We need to resample base point too if Hourly
                        if granularity == "Hourly":
                            bp_agg = df_comp.resample('h', on='Time')['Base_Point_MW'].mean()
                            fig_bench.add_trace(go.Scatter(x=bp_agg.index, y=bp_agg, name='Grid Limit (MW)', line=dict(color='red', width=1, dash='dot')))
                        else:
                            fig_bench.add_trace(go.Scatter(x=df_comp['Time'], y=df_comp['Base_Point_MW'], name='Grid Limit (MW)', line=dict(color='red', width=1, dash='dot')))
                
                fig_bench.update_layout(
                    title=f"Benchmarking: {res['resource_id']} ({res['start']} to {res['end']}) - {granularity}",
                    xaxis_title="Time",
                    yaxis_title=f"Generation ({unit})",
                    hovermode="x unified",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_bench, use_container_width=True)
            
            # Export Section
            st.markdown("#### 📥 Export Data")
            
            # Preview Table
            with st.expander("📋 View Data Table", expanded=False):
                st.dataframe(df_comp[['Time', 'Actual_MW', 'Modeled_MW']].head(1000), use_container_width=True)
            
            # Download Logic (Actual vs Modeled)
            export_cols = ['Time', 'Actual_MW', 'Modeled_MW']
            if 'Base_Point_MW' in df_comp.columns:
                export_cols.append('Base_Point_MW')
                
            csv_comp = df_comp[export_cols].to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label=f"📥 Download Comparison CSV (Actual vs Modeled)",
                data=csv_comp,
                file_name=f"benchmark_comp_{res['resource_id']}_{res['start']}_{res['end']}.csv",
                mime="text/csv",
                help="Includes Time, Actual MW, and Modeled MW for every 15-minute interval."
            )
            
            st.success(f"Successfully retrieved **{len(res['df_actual'])}** interval points.")

# --- Azure Sky Analysis Tab ---
# Azure Sky Analysis tab hidden for now
# with tab_azure_sky:
#     try:
#         tab_azure.render()
#     except Exception as e:
#         st.error(f"Error loading Azure Sky Analysis: {e}")
#         st.code(str(e))

with tab_vppa_8760:
    try:
        tab_vppa_8760_compare.render()
    except Exception as e:
        st.error(f"Error loading VPPA 8760 Compare: {e}")
        st.code(str(e))

st.markdown("---")
st.caption("Claude Project: [Open link](https://claude.ai/project/019c8810-9a99-7236-a14f-64b8e10f853a)")
