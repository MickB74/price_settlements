import streamlit as st
import os
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import gridstatus
import patch_gridstatus # Apply monkey patch
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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

# --- Constants & Configuration ---
HUB_LOCATIONS = {
    "HB_NORTH": (32.3865, -96.8475),   # Waxahachie, TX (I-35 solar corridor)
    "HB_SOUTH": (26.9070, -99.2715),   # Zapata, TX (South Texas inland wind belt - where projects actually are)
    "HB_WEST": (32.4518, -100.5371),   # Roscoe, TX ("Wind Energy Capital of Texas" - best West TX wind resource)
    "HB_HOUSTON": (29.3013, -94.7977), # Galveston, TX (Houston Hub's only wind project - excellent coastal wind resource)
    "HB_PAN": (35.2220, -101.8313),    # Amarillo, TX (Panhandle)
}

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

# --- State Management ---
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []



# Create Tabs
tab_scenarios, tab_validation, tab_performance = st.tabs(["Scenario Analysis", "Bill Validation", "Model Performance"])

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
                if ((tab.innerText.includes("Bill Validation") || tab.innerText.includes("Model Performance")) && tab.getAttribute("aria-selected") === "true") {
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

with tab_scenarios:
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



# --- Data Fetching ---
# --- Data Fetching ---
@st.cache_data(show_spinner="Fetching data from ERCOT (this may take 1-2 minutes for the first load)...")
def get_ercot_data(year, _mtime=None):
    """Fetches and caches ERCOT RTM data for a given year. _mtime used for cache invalidation."""
    # Local cache file path
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
        st.error(f"Error fetching data for {year}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_cached_asset_data(resource_id, start_date, end_date):
    """Cached wrapper for SCED fetching to prevent re-loading on every interaction."""
    return sced_fetcher.get_asset_period_data(resource_id, start_date, end_date)

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
            df_hub = df_hub[df_hub['Time_Central'].dt.month == target_month].copy()
    
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
                # File missing?
                potential_gen = np.zeros(len(df_hub))
                profile_central = None

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
                force_tmy=force_tmy
            )
            # Align profile with df_hub timestamps
            profile_central = profile_series.tz_convert('US/Central')
        
        if profile_central is not None:
             # Reindex to match df_hub['Time_Central']
             potential_gen = profile_central.reindex(df_hub['Time_Central'], method='nearest').values
        else:
             potential_gen = np.zeros(len(df_hub))
        
    except Exception as e:
        try:
            # Fallback if strict reindex fails due to tz issues
             potential_gen = np.zeros(len(df_hub))
        except:
             potential_gen = np.zeros(len(df_hub))

    df_hub['Potential_Gen_MW'] = potential_gen
    
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
    
    # --- Cover Page ---
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
    
    # --- Footer ---
    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph("<i>Generated by VPPA Settlement Estimator</i>", normal_style))
    
    # Build PDF
    doc.build(elements)
    
    # Return buffer
    pdf_buffer.seek(0)
    return pdf_buffer

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

# --- Sidebar: Scenario Builder ---
st.sidebar.header("Scenario Builder")


# --- Solar/Wind (Batch) Mode ---
# --- Map Location Picker (outside form) ---
with st.sidebar.expander("🗺️ Pick Location on Map", expanded=False):
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
    map_data = st_folium(m, height=300, width=280, returned_objects=["last_clicked"])
    
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
    
    # Calculate and suggest nearest hub
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

# --- Solar/Wind Batch Form ---
with st.sidebar.form("add_scenario_form"):
    st.subheader("Add Scenarios")
    
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
    
    # Custom Location Override
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
    
    st.markdown("---")
    
    # Two buttons: Add (append) vs Clear & Run (reset)
    # Three buttons: Add (append), Clear & Run (replace), Reset All (clear)
    # Button Layout: 
    # Row 1: Add (Primary Action)
    add_button = st.form_submit_button("➕ Add Scenarios", type="primary", use_container_width=True)
    
    # Row 2: Secondary Actions
    col_clear, col_reset = st.columns(2)
    with col_clear:
        clear_run_button = st.form_submit_button("🏃 Run", type="secondary", use_container_width=True)
    with col_reset:
        reset_all_button = st.form_submit_button("🗑️ Reset", type="secondary", use_container_width=True, on_click=reset_defaults)
    
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
                            
                            if s_use_custom_location and s_custom_lat is not None:
                                name += f" [Custom: {s_custom_lat:.2f}, {s_custom_lon:.2f}]"
                                
                            # Check for duplicates
                            if any(s['name'] == name for s in st.session_state.scenarios):
                                continue 
                            else:
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
                                    "custom_lat": s_custom_lat if s_use_custom_location else None,
                                    "custom_lon": s_custom_lon if s_use_custom_location else None,
                                    "custom_profile_path": None
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
                            
                            if s_use_custom_location and s_custom_lat is not None:
                                name += f" [Custom: {s_custom_lat:.2f}, {s_custom_lon:.2f}]"
                                
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
                                "custom_lat": s_custom_lat if s_use_custom_location else None,
                                "custom_lon": s_custom_lon if s_use_custom_location else None,
                                "custom_profile_path": None
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

        # Calculate Results
        results = []
        progress_bar = st.progress(0)

        for i, scenario in enumerate(st.session_state.scenarios):
            # Fetch Data
            cache_path = f"ercot_rtm_{scenario['year']}.parquet"
            mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0
            df_rtm = get_ercot_data(scenario['year'], _mtime=mtime)
            if df_rtm.empty:
                st.warning(f"Could not fetch data for {scenario['name']}")
                continue
        
            # Calculate
            df_res = calculate_scenario(scenario, df_rtm)
    
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
                f"**\${best_val:,.0f}**, while **{worst_scen}** trails at **\${worst_val:,.0f}**."
            )
        else:
            st.markdown(
                f"**Insight:** The **{best_scen}** scenario has a total settlement of **\${best_val:,.0f}**."
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
                    f"**Insight:** **{best_scen}** led with a total settlement of **\${best_val:,.0f}**."
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
        
                st.markdown(
                    f"**Insight:** The highest monthly return was **\${best_month_row['Settlement_Amount']:,.0f}** "
                    f"in **{best_month_row['Month_Date'].strftime('%B %Y')}** ({best_month_row['Scenario']}), "
                    f"whereas the lowest was **\${worst_month_row['Settlement_Amount']:,.0f}** "
                    f"in **{worst_month_row['Month_Date'].strftime('%B %Y')}** ({worst_month_row['Scenario']})."
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
                cache_path = f"ercot_rtm_{year_val}.parquet"
                mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0
                df_rtm = get_ercot_data(year_val, _mtime=mtime)
                if not df_rtm.empty:
                    df_display = calculate_scenario(selected_scenario_config, df_rtm)
            
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
                if st.button("Prepare Summary Excel"):
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        df_summary.to_excel(writer, sheet_name='Summary', index=False)
                        if monthly_data:
                            df_monthly.to_excel(writer, sheet_name='Monthly Details', index=False)
                        # We could add daily aggregates too if useful
            
                    st.download_button(
                        label="Download Summary Excel",
                        data=excel_buffer.getvalue(),
                        file_name="vppa_summary_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
                    st.info("Note: Detailed ZIP download is disabled to save memory. Use 'View Raw Data' to download specific scenario CSVs.")
        
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


# --- Bill Validation Tab ---
with tab_validation:
    st.header("Bill Validation")
    st.markdown("Upload your generation data (and optional settlement data) to validate against official ERCOT market prices.")

    # --- Configuration & Preview Header ---
    st.subheader("⚙️ Analysis Configuration")
    st.caption("Configure market parameters and project profiles for validation")

    # --- Integrated Configuration & Preview Controls ---
    with st.container():
        if 'val_custom_lat' not in st.session_state: st.session_state.val_custom_lat = 32.0
        if 'val_custom_lon' not in st.session_state: st.session_state.val_custom_lon = -100.0

        def update_loc_from_hub():
            # Only update if custom location is NOT checked (acting as a lock)
            if not st.session_state.get('val_use_custom_location', False):
                h_lat, h_lon = HUB_LOCATIONS.get(st.session_state.val_hub, (32.0, -100.0))
                st.session_state.val_custom_lat = h_lat
                st.session_state.val_custom_lon = h_lon
                st.session_state.val_map_lat = h_lat
                st.session_state.val_map_lon = h_lon

        # Row 1: Hub & Year & Price
        # c1, c2, c3, c4 = st.columns(4)
        c0, c1, c2, c3, c4 = st.columns([0.1, 1, 1, 1, 1]) # spacer
        with c1:
            # Hub Selection with Callback
            val_hub = st.selectbox("Select Hub", list(HUB_LOCATIONS.keys()), key="val_hub", on_change=update_loc_from_hub)
            # Update map location if not using custom location
            if not st.session_state.get('val_use_custom_location', False):
                lat, lon = HUB_LOCATIONS.get(val_hub, (32.0, -100.0))
                # Update map session state if not custom
                st.session_state.val_map_lat = lat
                st.session_state.val_map_lon = lon
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

        # Row 2: Technology & Preview Settings + Action Button
        # Row 2: Technology & Preview Settings
        c5, c6, c7 = st.columns(3)
        with c5:
            preview_tech = st.selectbox("Technology", ["Solar", "Wind"], key="preview_tech")
        with c6:
            preview_capacity = st.number_input("Capacity (MW)", min_value=1.0, max_value=1000.0, value=100.0, step=10.0, key="preview_capacity")
        with c7:
            preview_weather = st.selectbox("Weather Source", ["Actual Weather", "Typical Year (TMY)", "Compare Both"], key="preview_weather")
            
        # Optional Turbine Selector (if Wind)
        selected_turbine = "GENERIC"
        if preview_tech == "Wind":
            turbine_opts = ["Generic (IEC Class 2)", "Vestas V163 (Low Wind)", "GE 2.x (Workhorse)", "GE 3.6-154 (Modern Mainstream)", "Nordex N163 (5.X MW)"]
            c_turb1, c_turb2, c_turb3 = st.columns(3)
            with c_turb1:
                val_turb_ui = st.selectbox("Turbine Model", turbine_opts, key="val_preview_turbine")
            
            turbine_override_map = {
                "Generic (IEC Class 2)": "GENERIC",
                "Vestas V163 (Low Wind)": "VESTAS_V163",
                "GE 2.x (Workhorse)": "GE_2X",
                "GE 3.6-154 (Modern Mainstream)": "GE_3X",
                "Nordex N163 (5.X MW)": "NORDEX_N163"
            }
            if val_turb_ui in turbine_override_map:
                selected_turbine = turbine_override_map[val_turb_ui]

        # Row 3: Actions
        c8, c9 = st.columns([3, 1])
        with c8:
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            curtail_neg = st.checkbox("Curtail when Price < $0", value=False, help="Set Generation to 0 MWh when Hub Price is negative")
        with c9:
             if st.button("📈 Generate Preview", type="primary", use_container_width=True):
                with st.spinner("Generating profile..."):
                    try:
                        # Market Data
                        cache_path = f"ercot_rtm_{val_year}.parquet"
                        mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0
                        
                        df_market = get_ercot_data(val_year, _mtime=mtime)
                        if df_market.empty:
                            st.error(f"No market data for {val_year}")
                        else:
                            df_market_hub = df_market[df_market['Location'] == val_hub].copy()
                            
                            # Location handling
                            # Location handling - Always use custom/synced lat/lon
                            lat, lon = st.session_state.val_custom_lat, st.session_state.val_custom_lon
                            
                            weather_opts = []
                            if preview_weather == "Actual Weather": weather_opts = [{"name": "Actual", "force_tmy": False}]
                            elif preview_weather == "Typical Year (TMY)": weather_opts = [{"name": "TMY", "force_tmy": True}]
                            else: weather_opts = [{"name": "Actual", "force_tmy": False}, {"name": "TMY", "force_tmy": True}]
                            
                            preview_results = {}
                            for source in weather_opts:
                                profile = fetch_tmy.get_profile_for_year(year=val_year, tech=preview_tech, lat=lat, lon=lon, capacity_mw=preview_capacity, force_tmy=source["force_tmy"], turbine_type=selected_turbine)
                                if profile is not None:
                                    pc = profile.tz_convert('US/Central')
                                    pdf = pd.DataFrame({'Gen_MW': pc.values, 'Time': pc.index.tz_convert('UTC')})
                                    pdf['Gen_Energy_MWh'] = pdf['Gen_MW'] * 0.25
                                    merged = pd.merge(pdf, df_market_hub[['Time', 'SPP', 'Time_Central']], on='Time', how='inner')
                                    
                                    # Filter by selected months
                                    m_list = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
                                    sel_m_nums = [i+1 for i, m in enumerate(m_list) if st.session_state.get(f"val_sel_{m}", True)]
                                    if sel_m_nums:
                                        merged = merged[merged['Time_Central'].dt.month.isin(sel_m_nums)].copy()
                                    
                                    # Calculate Potential Curtailment (Always)
                                    merged['Potential_Curtailed_MWh'] = 0.0
                                    mask_neg_price = merged['SPP'] < 0
                                    if not merged.empty:
                                        merged.loc[mask_neg_price, 'Potential_Curtailed_MWh'] = merged.loc[mask_neg_price, 'Gen_Energy_MWh']

                                    merged['Curtailed_MWh'] = 0.0
                                    if not merged.empty:
                                        # Apply Curtailment if selected
                                        if curtail_neg:
                                            merged.loc[mask_neg_price, 'Curtailed_MWh'] = merged.loc[mask_neg_price, 'Gen_Energy_MWh']
                                            merged.loc[mask_neg_price, 'Gen_Energy_MWh'] = 0
                                        else:
                                            # If not curtailed, 'Curtailed_MWh' is 0, but we have 'Potential' stored
                                            pass
                                        
                                        rs_pct = val_revenue_share / 100.0
                                        settle_p = (np.maximum(merged['SPP'] - val_vppa_price, 0) * rs_pct) + np.minimum(merged['SPP'] - val_vppa_price, 0) if rs_pct < 1.0 else merged['SPP'] - val_vppa_price
                                        merged['Settlement_$/MWh'] = settle_p
                                        merged['Settlement_$'] = merged['Gen_Energy_MWh'] * settle_p
                                        merged['Market_Revenue_$'] = merged['Gen_Energy_MWh'] * merged['SPP']
                                        merged['VPPA_Payment_$'] = merged['Gen_Energy_MWh'] * val_vppa_price
                                        preview_results[source["name"]] = merged
                            
                            if preview_results:
                                st.session_state['val_preview_results'] = preview_results
                                st.session_state['val_preview_tech'] = preview_tech
                            else:
                                st.error("No data generated for criteria.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # --- Month Selection ---
    with st.expander("📅 Select Months for Validation", expanded=True):
        st.caption("Select which months to include in the validation analysis")
        
        val_months = ["January", "February", "March", "April", "May", "June", 
                      "July", "August", "September", "October", "November", "December"]
        
        # Select All / Clear All buttons
        col_m1, col_m2 = st.columns([0.2, 0.8])
        if col_m1.button("Select All", key="val_all_months"):
            for m in val_months:
                st.session_state[f"val_sel_{m}"] = True
        if col_m2.button("Clear All", key="val_clear_months"):
            for m in val_months:
                st.session_state[f"val_sel_{m}"] = False

        selected_month_names = []
        cols = st.columns(4)
        for i, month in enumerate(val_months):
            with cols[i % 4]:
                if st.checkbox(month, value=st.session_state.get(f"val_sel_{month}", True), key=f"val_sel_{month}"):
                    selected_month_names.append(month)
        
        # Convert month names to numbers (1-12)
        month_map = {m: i+1 for i, m in enumerate(val_months)}
        selected_month_numbers = [month_map[m] for m in selected_month_names]
        
        if not selected_month_numbers:
            st.warning("⚠️ No months selected. Please select at least one month.")
    
    # Custom Location Toggle and Manual Input
    # Custom Location Toggle and Manual Input
    val_use_custom_location = st.checkbox("Use Custom Project Location", value=False, help="Specify exact project coordinates", key="val_use_custom_location")
    
    st.caption("💡 Enter your project's exact coordinates or use the map below")
    col_lat, col_lon = st.columns(2)
    with col_lat:
        val_custom_lat = st.number_input("Latitude", min_value=25.0, max_value=40.0, step=0.01, format="%.4f", key="val_custom_lat")
    with col_lon:
        val_custom_lon = st.number_input("Longitude", min_value=-107.0, max_value=-93.0, step=0.01, format="%.4f", key="val_custom_lon")
    
    st.info(f"📍 Selected location: {val_custom_lat:.4f}, {val_custom_lon:.4f}")

    # --- Location Picker Section ---
    with st.expander("🗺️ Pick Project Location", expanded=False):
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
                    # Auto-check the "Use Custom Location" checkbox
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
            # Also sync to form input keys so they update
            st.session_state.val_custom_lat = clicked_lat
            st.session_state.val_custom_lon = clicked_lon
            # Auto-check the "Use Custom Location" checkbox
            st.session_state.val_use_custom_location = True
            
            # Calculate nearest hub on click
            def calc_dist(lat1, lon1, lat2, lon2):
                return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
            click_distances = {hub: calc_dist(clicked_lat, clicked_lon, lat, lon) for hub, (lat, lon) in hub_locations.items()}
            nearest_hub = min(click_distances, key=click_distances.get)
            st.info(f"📍 Location set to: {clicked_lat:.4f}, {clicked_lon:.4f} | Nearest hub: **{nearest_hub}** (select from dropdown above)")
        
        # Show current selected coordinates
        if st.session_state.get('val_use_custom_location', False):
            st.success(f"✅ Using custom location: {st.session_state.val_map_lat:.4f}, {st.session_state.val_map_lon:.4f}")
    


    
    # --- Display Results Section (Cached) ---
    if 'val_preview_results' in st.session_state:
        preview_results = st.session_state['val_preview_results']
        
        # 1. Comparison Table if multiple sources
        if len(preview_results) > 1:
            st.markdown("### 📊 Contrast: Actual vs Typical (TMY)")
            comp_summary = []
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
            st.table(pd.DataFrame(comp_summary))
            st.markdown("---")
        
        # 2. Main Metrics (show either first one or Actual if available)
        primary_name = "Actual" if "Actual" in preview_results else list(preview_results.keys())[0]
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
        
        # 3. Charts with own View Selector
        st.markdown("---")
        chart_col, view_col = st.columns([0.7, 0.3])
        with chart_col:
            st.markdown("### 📊 Settlement Chart")
        with view_col:
            preview_view = st.selectbox("Chart Time Aggregation", ["Daily", "Monthly"], key="preview_view_internal")

        fig = go.Figure()
        
        for name, df in preview_results.items():
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
        
        df_primary['Implied_REC_Cost_$/MWh'] = -(df_primary['Settlement_$'] / df_primary['Gen_Energy_MWh']).fillna(0)
        
        display_cols = ['Time_Central', 'Gen_MW', 'Gen_Energy_MWh', 'SPP', 'Settlement_$/MWh', 'Settlement_$', 'Implied_REC_Cost_$/MWh']
        preview_df = df_primary[display_cols].head(100).copy()
        preview_df['Time_Central'] = preview_df['Time_Central'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(
            preview_df.style.format({
                'Gen_MW': '{:.2f}',
                'Gen_Energy_MWh': '{:.4f}',
                'SPP': '${:.2f}',
                'Settlement_$/MWh': '${:.2f}',
                'Settlement_$': '${:.2f}',
                'Implied_REC_Cost_$/MWh': '${:.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Download full dataset
        csv = df_primary[['Time_Central'] + display_cols[1:]].to_csv(index=False).encode('utf-8')
        p_tech = st.session_state.get('val_preview_tech', 'profile')
        st.download_button(
            label=f"📥 Download Full {primary_name} Intervals CSV",
            data=csv,
            file_name=f"{p_tech}_{val_hub}_{val_year}_{primary_name}_intervals.csv",
            mime="text/csv",
            key="download_preview_csv"
        )

        # Download PDF Bill
        pdf_config = {
            'hub': st.session_state.get('val_hub', val_hub),
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
            file_extension = uploaded_bill.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df_bill = pd.read_csv(uploaded_bill)
            elif file_extension in ['xlsx', 'xls']:
                df_bill = pd.read_excel(uploaded_bill, engine='openpyxl' if file_extension == 'xlsx' else None)
            elif file_extension == 'pdf':
                # PDF parsing - extract tables using pdfplumber
                import pdfplumber
                
                with pdfplumber.open(uploaded_bill) as pdf:
                    # Try to extract tables from all pages
                    tables = []
                    for page in pdf.pages:
                        page_tables = page.extract_tables()
                        if page_tables:
                            tables.extend(page_tables)
                    
                    if not tables:
                        st.error("❌ No tables found in PDF. Please ensure your PDF contains tabular data.")
                        st.stop()
                    
                    # Use the first table found (or combine if multiple)
                    # Assume first row is header
                    if len(tables) == 1:
                        table_data = tables[0]
                        headers = table_data[0]
                        rows = table_data[1:]
                        df_bill = pd.DataFrame(rows, columns=headers)
                    else:
                        # Multiple tables - try to combine or use largest
                        st.info(f"ℹ️ Found {len(tables)} tables in PDF. Using the largest table.")
                        largest_table = max(tables, key=len)
                        headers = largest_table[0]
                        rows = largest_table[1:]
                        df_bill = pd.DataFrame(rows, columns=headers)
            else:
                st.error(f"Unsupported file type: {file_extension}")
                st.stop()
            
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
                # Parse Time
                # Parse Time - Localize to Central if naive
                df_bill['Time'] = pd.to_datetime(df_bill[time_col])
                if df_bill['Time'].dt.tz is None:
                    df_bill['Time'] = df_bill['Time'].dt.tz_localize('US/Central', ambiguous='infer').dt.tz_convert('UTC')
                else:
                    df_bill['Time'] = df_bill['Time'].dt.tz_convert('UTC')
                
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
                    
                    # Process User Bill Data (ensure it has Month Period)
                    if 'Month' not in df_bill.columns and time_col:
                        df_bill['MonthPeriod'] = pd.to_datetime(df_bill[time_col]).dt.to_period('M')
                        df_bill['Month'] = df_bill['MonthPeriod'].dt.strftime('%b %Y')
                    elif 'Month' not in df_bill.columns:
                        # Fallback if no time column (unlikely given previous checks)
                        st.error("Cannot align data: Missing time column in bill.")
                        st.stop()
                        
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
                        time_diff = pd.to_datetime(df_bill[time_col]).diff().median()
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
                    
                    # 3. Fetch Market Data for the entire year
                    cache_path = f"ercot_rtm_{val_year}.parquet"
                    mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0
                    df_market = get_ercot_data(val_year, _mtime=mtime)
                    
                    if df_market.empty:
                        st.error(f"Could not find market data for {val_year}.")
                    else:
                        # Filter for selected Hub
                        df_market_hub = df_market[df_market['Location'] == val_hub].copy()
                        
                        # Parse user's time column to get months
                        df_bill['Month'] = pd.to_datetime(df_bill[time_col]).dt.to_period('M')
                        
                        # Calculate expected settlement for the entire year using market data
                        # We'll need generation profile - use user's total gen and distribute it
                        # Actually, we just need total MWh per month from user, then calc settlement
                        
                        st.subheader("Monthly Breakdown")
                        
                        comparison_data = []
                        
                        for _, row in df_bill.iterrows():
                            month_period = row['Month']
                            
                            # Skip months not selected by the user
                            if selected_month_numbers and month_period.month not in selected_month_numbers:
                                continue
                                
                            user_gen_mwh = row['User_Gen_MW']  # Assuming this is MWh total for the month
                            user_settlement = row.get('User_Settlement_Amount', 0)
                            
                            # Filter market data for this month
                            month_start = month_period.to_timestamp().tz_localize('US/Central')
                            month_end = (month_period + 1).to_timestamp().tz_localize('US/Central')
                            
                            df_month = df_market_hub[
                                (df_market_hub['Time_Central'] >= month_start) & 
                                (df_market_hub['Time_Central'] < month_end)
                            ].copy()
                            
                            if not df_month.empty:
                                # Calculate expected settlement with revenue sharing
                                avg_market_price = df_month['SPP'].mean()
                                
                                # Apply revenue share logic
                                revenue_share_pct = val_revenue_share / 100.0
                                if revenue_share_pct < 1.0:
                                    # When SPP > VPPA: Settlement = (SPP - VPPA) * share_pct (buyer gets only their share of upside)
                                    # When SPP <= VPPA: Settlement = SPP - VPPA (full downside, no sharing)
                                    upside = max(avg_market_price - val_vppa_price, 0)
                                    downside = min(avg_market_price - val_vppa_price, 0)
                                    settlement_per_mwh = (upside * revenue_share_pct) + downside
                                else:
                                    settlement_per_mwh = avg_market_price - val_vppa_price
                                
                                expected_settlement = user_gen_mwh * settlement_per_mwh
                                
                                comparison_data.append({
                                    'Month': month_period.strftime('%b %Y'),
                                    'Generation (MWh)': user_gen_mwh,
                                    'Avg Market Price': f"${avg_market_price:.2f}",
                                    'Strike Price': f"${val_vppa_price:.2f}",
                                    'Calculated Settlement': expected_settlement,
                                    'User Reported': user_settlement,
                                    'Difference': expected_settlement - user_settlement
                                })
                        
                        df_comparison = pd.DataFrame(comparison_data)
                        
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
                                2. We calculated the average market price at **{val_hub}** for that month using ERCOT RTM data
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
                                2. We calculated the average market price at **{val_hub}** for that month using ERCOT RTM data
                                3. Settlement = Generation × (Avg Market Price - Strike Price)
                                
                                **Formula:**
                                ```
                                Monthly Settlement = Generation (MWh) × (Avg SPP - ${val_vppa_price:.2f})
                                ```
                                
                                **Note:** This uses monthly average pricing. For precise validation, interval-level data is recommended.
                                """)
                
                else:
                    # Original interval-level validation logic
                    # 3. Fetch Market Data
                    cache_path = f"ercot_rtm_{val_year}.parquet"
                    mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0
                    df_market = get_ercot_data(val_year, _mtime=mtime)
                
                    if df_market.empty:
                        st.error(f"Could not find market data for {val_year}.")
                    else:
                        # Filter for selected Hub
                        df_market_hub = df_market[df_market['Location'] == val_hub].copy()
                        
                        # 4. Merge Data
                        # Ensure timezone alignment - Market data is in UTC (Time) and Central (Time_Central)
                        # We'll match on UTC 'Time' for safety if user provided UTC, or convert user to UTC if naive
                        
                        # Handle User Timezone
                        if df_bill['Time'].dt.tz is None:
                            st.warning("⚠️ User timestamp is timezone-naive. Assuming US/Central.")
                            df_bill['Time'] = df_bill['Time'].dt.tz_localize('US/Central', ambiguous='infer').dt.tz_convert('UTC')
                        else:
                            df_bill['Time'] = df_bill['Time'].dt.tz_convert('UTC')
                        
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
        st.subheader("Wind Model Benchmarking (Q4 2024)")
        
        # Metrics Overview
        # Filter for Advanced model
        wind_advanced = wind_res[wind_res['Model'].str.contains('Advanced')]
        avg_r_wind = wind_advanced['R'].mean()
        max_r_wind = wind_advanced['R'].max()
        
        m1, m2 = st.columns(2)
        m1.metric("Avg Correlation (R)", f"{avg_r_wind:.2f}", help="Correlation between synthetic and actual generation")
        m2.metric("Top Correlation", f"{max_r_wind:.2f}")

        st.markdown("### 🏆 Wind Leaderboard (Advanced Model)")
        top_wind = wind_advanced.sort_values('R', ascending=False).head(10)
        st.dataframe(
            top_wind[['Project', 'R', 'MBE (MW)', 'RMSE (MW)']],
            column_config={
                "R": st.column_config.NumberColumn(
                    "Correlation (R)",
                    help="Pearson Correlation Coefficient. Measures how well the shape of the modeled profile matches actual generation. (1.0 = Perfect match)",
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
        st.subheader("Solar Model Benchmarking (Q4 2024)")
        
        # Metrics Overview
        solar_advanced = solar_res[solar_res['Model'].str.contains('Advanced')]
        avg_r_solar = solar_advanced['R'].mean()
        max_r_solar = solar_advanced['R'].max()
        
        m1, m2 = st.columns(2)
        m1.metric("Avg Correlation (R)", f"{avg_r_solar:.2f}")
        m2.metric("Top Correlation", f"{max_r_solar:.2f}")

        st.markdown("### 🏆 Solar Leaderboard (Tracking Model)")
        top_solar = solar_advanced.sort_values('R', ascending=False).head(10)
        st.dataframe(
            top_solar[['Project', 'R', 'MBE (MW)', 'RMSE (MW)']],
            column_config={
                "R": st.column_config.NumberColumn(
                    "Correlation (R)",
                    help="Pearson Correlation Coefficient. Measures how well the shape of the modeled profile matches actual generation. (1.0 = Perfect match)",
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
        benchmark_asset_name = st.selectbox("Select Curated Project", options=["None (Use Custom ID)"] + list(asset_registry.keys()), index=1)
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
        with st.expander("🛠️ Advanced Model Parameters"):
            c_p1, c_p2, c_p3 = st.columns(3)
            # Losses
            model_losses = c_p1.slider("System Losses (%)", 0, 50, 14, help="Reduces gross modeled output (Wake, Electrical, Availability). Default ~14%.")
            
            # Bias Correction
            model_bias = c_p2.number_input("Linear Bias Correction (Multiplier)", 0.5, 1.5, 1.0, step=0.01, help="Scalar to linearly tune model up/down.")
            
            # Turbine Type
            turbine_opts = ["Auto-Detect", "Generic (IEC Class 2)", "Vestas V163 (Low Wind)", "GE 2.x (Workhorse)", "GE 3.6-154 (Modern Mainstream)", "Nordex N163 (5.X MW)"]
            selected_turb = c_p3.selectbox("Turbine Type Override", turbine_opts)
            
            turbine_override_map = {
                "Auto-Detect": None,
                "Generic (IEC Class 2)": "GENERIC",
                "Vestas V163 (Low Wind)": "VESTAS_V163",
                "GE 2.x (Workhorse)": "GE_2X",
                "GE 3.6-154 (Modern Mainstream)": "GE_3X",
                "Nordex N163 (5.X MW)": "NORDEX_N163"
            }
            final_turbine_req = turbine_override_map[selected_turb]

        if st.button(f"🚀 Fetch & Benchmark {final_resource_id}"):
            if start_bench > end_bench:
                st.error("Start date must be before end date.")
            else:
                with st.spinner(f"Retrieving actual production and running model for {final_resource_id}..."):
                    # 1. Fetch Actual Data
                    df_actual = get_cached_asset_data(final_resource_id, start_bench, end_bench)
                    
                    if not df_actual.empty:
                        # 2. Run Model for comparison
                        # Use metadata if available, otherwise use page settings
                        compare_lat = asset_meta['lat'] if asset_meta else 32.4487
                        compare_lon = asset_meta['lon'] if asset_meta else -99.7331
                        compare_tech = asset_meta['tech'] if asset_meta else "Solar"
                        compare_cap = asset_meta['capacity_mw'] if asset_meta else 100.0
                        
                        # Use enriched model first, then manual type, then generic
                        # If override is set, use it.
                        if final_turbine_req:
                            compare_turbine = final_turbine_req
                        else:
                            compare_turbine = asset_meta.get('turbine_model', asset_meta.get('turbine_type', 'GENERIC')) if asset_meta else 'GENERIC'
                        
                        # Run model for all years in range
                        target_years = sorted(list(set([d.year for d in pd.to_datetime(df_actual['Time'])])))
                        model_dfs = []
                        for yr in target_years:
                            m_df = fetch_tmy.get_profile_for_year(yr, compare_tech, compare_cap, compare_lat, compare_lon, turbine_type=compare_turbine)
                            model_dfs.append(m_df)
                        df_modeled_full = pd.concat(model_dfs)
                        
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
                            'turbine': compare_turbine
                        }
                    else:
                        st.error(f"No generation data found for `{final_resource_id}` in this period. Note: Data stops ~60 days before today.")
        
        # --- Visualization Section (Outside Button Logic) ---
        if 'bench_results' in st.session_state:
            res = st.session_state['bench_results']
            
            st.divider()
            st.markdown(f"### 📊 Results for `{res['resource_id']}`")
            
            df_comp = res['df_comp']
            
            # Time Granularity Selector
            st.markdown("#### ⏱️ Time Resolution")
            granularity = st.radio("Select View:", ["15-Minute", "Hourly", "Daily", "Monthly", "Annual"], horizontal=True, index=0)
                        
            # Prepare data for aggregation
            df_comp['Actual_MWh'] = df_comp['Actual_MW'] / 4.0
            df_comp['Modeled_MWh'] = df_comp['Modeled_MW'] / 4.0
            
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
                agg_actual_sum = df_agg[y_col_act].sum()
                agg_modeled_sum = df_agg[y_col_mod].sum()
                
                mae = (df_agg[y_col_act] - df_agg[y_col_mod]).abs().mean()
                r2 = np.corrcoef(df_agg[y_col_act], df_agg[y_col_mod])[0, 1]**2 if len(df_agg) > 1 else 0
                
                # Bias / % Diff
                diff_pct = ((agg_modeled_sum - agg_actual_sum) / agg_actual_sum) if agg_actual_sum > 0 else 0
                
                # Display Metrics row 1: Totals (Scale Independent essentially, apart from small resampling diffs)
                st.markdown("#### 📊 Numerical Summary")
                mc1, mc2, mc3 = st.columns(3)
                
                # Display Totals in MWh (always useful) or Avg MW depending on view? 
                # Let's keep totals as MWh for clarity on volume
                total_actual_mwh = df_comp['Actual_MWh'].sum()
                total_modeled_mwh = df_comp['Modeled_MWh'].sum()
                
                mc1.metric("Total Actual Gen", f"{total_actual_mwh:,.1f} MWh")
                mc2.metric("Total Model Est", f"{total_modeled_mwh:,.1f} MWh", delta=f"{diff_pct:+.1%}", delta_color="inverse")
                mc3.metric("Total Bias", f"{total_modeled_mwh - total_actual_mwh:+.1f} MWh")
                
                # Display Metrics row 2: Statistical Fit (Context Dependent)
                m1, m2, m3 = st.columns(3)
                m1.metric(f"Mean Abs Error ({unit})", f"{mae:.1f}")
                m2.metric("Correlation (R²)", f"{r2:.2%}")
                m3.metric("Data Points", f"{len(df_agg):,}")
                
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
            csv_actual = res['df_actual'][['Time', 'Actual_MW']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download Actual Generation CSV ({res['resource_id']})",
                data=csv_actual,
                file_name=f"actual_gen_{res['resource_id']}_{res['start']}_{res['end']}.csv",
                mime="text/csv"
            )
            
            st.success(f"Successfully retrieved **{len(res['df_actual'])}** interval points.")
