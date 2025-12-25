import streamlit as st
import pandas as pd
import numpy as np
import gridstatus
import patch_gridstatus # Apply monkey patch
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
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

# Page Config
st.set_page_config(page_title="VPPA Settlement Estimator", layout="wide")

# Title
st.title("VPPA Settlement Estimator")
st.markdown("Compare multiple Virtual Power Purchase Agreement (VPPA) scenarios in ERCOT.")

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
    - **Coverage:** 2020-2025 (historical actual prices)
    - **Hubs:** HB_NORTH, HB_SOUTH, HB_WEST, HB_HOUSTON, HB_PAN
    
    ### 2. **Generation Profiles (Wind/Solar)**
    - **Sources:** 
        - **[Open-Meteo](https://open-meteo.com/):** For **2024 Actuals** (ERA5 Reanalysis). High-accuracy solar irradiance and 100m wind speeds.
        - **[PVGIS](https://re.jrc.ec.europa.eu/pvgis/):** For **History (2005-2023)** and **Typical Meteorological Year (TMY)** data.
    - **Method:**
      - **2024:** Uses **Actual Open-Meteo Data** (Solar & Wind) ✅
      - **Historical Years:** Uses **Actual PVGIS Data**
      - **Future/TMY:** Uses **TMY Data** (Typical Meteorological Year) representing long-term averages.
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
    1. **Fetch Weather Data** from Open-Meteo (2024) or PVGIS (History/TMY)
    2. **Convert to Power:**
       - Solar: GHI (Global Horizontal Irradiance) → DC power → inverter efficiency → AC MW
       - Wind: Wind speed (scaled to hub height) → power curve → MW
    3. **Resample** to 15-minute intervals
    4. **Align** timestamps to ERCOT Central Time
    
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


# --- State Management ---
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []

# --- Data Fetching ---
# --- Data Fetching ---
@st.cache_data(show_spinner="Fetching data from ERCOT (this may take 1-2 minutes for the first load)...")
def get_ercot_data(year):
    """Fetches and caches ERCOT RTM data for a given year."""
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

    # Hub Locations (Lat, Lon)
    # Updated based on actual renewable project concentrations from ERCOT queue data
    HUB_LOCATIONS = {
        "HB_NORTH": (32.3865, -96.8475),   # Waxahachie, TX (I-35 solar corridor)
        "HB_SOUTH": (26.9070, -99.2715),   # Zapata, TX (South Texas inland wind belt - where projects actually are)
        "HB_WEST": (32.4518, -100.5371),   # Roscoe, TX ("Wind Energy Capital of Texas" - best West TX wind resource)
        "HB_HOUSTON": (29.3013, -94.7977), # Galveston, TX (Houston Hub's only wind project - excellent coastal wind resource)
        "HB_PAN": (35.2220, -101.8313),    # Amarillo, TX (Panhandle)
    }
    
    # Default to Abilene if hub not found
    default_loc = (32.4487, -99.7331)  # Abilene, TX
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
st.sidebar.header("Scenario Builder")

# Mode Selection (outside form)
mode = st.sidebar.radio("Mode", ["Solar/Wind (Batch)", "Custom Upload"], index=0)

if mode == "Custom Upload":
    # --- Custom Upload Section (No Form) ---
    st.sidebar.subheader("Custom Upload")
    st.sidebar.markdown("*Upload a file to auto-create scenario*")
    
    available_years = [2025, 2024, 2023, 2022, 2021, 2020]
    common_hubs = ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON", "HB_PAN"]
    
    # Add "All Years" option to the dropdown
    year_options = ["All Years"] + available_years
    custom_year_selection = st.sidebar.selectbox("Year", year_options, index=0)
    
    # If "All Years" is selected, use all available years; otherwise use the selected year
    if custom_year_selection == "All Years":
        custom_years = available_years
    else:
        custom_years = [custom_year_selection]
    custom_hub = st.sidebar.selectbox("Hub", common_hubs, index=0)
    custom_capacity = st.sidebar.number_input("Capacity (MW)", value=80.0, step=10.0, key="custom_capacity")
    custom_vppa_price = st.sidebar.number_input("VPPA Price ($/MWh)", value=50.0, step=1.0, key="custom_vppa")
    custom_no_curtailment = st.sidebar.checkbox("Remove $0 floor (No Curtailment)", key="custom_curtail")
    
    st.sidebar.info("Upload a CSV file with your generation profile.")
    with st.sidebar.expander("File Format Guidance"):
        st.markdown("""
        **Required Format:** CSV file
        
        **Columns:**
        - `Gen_MW` (Required): Generation in MW.
        - `Time` (Optional): Datetime column.
        
        **Notes:**
        - If `Time` is missing, we assume data starts Jan 1st of the selected year.
        - **Hourly Data**: 8,760 rows (regular year) or 8,784 rows (leap year)
        - **15-Minute Data**: 35,040 rows (regular year) or 35,136 rows (leap year)
        """)
    
    uploaded_file = st.sidebar.file_uploader("Upload Profile (CSV)", type=["csv"], key="custom_uploader")
    
    # Auto-process on upload
    if uploaded_file is not None:
        import os
        
        # Ensure directory exists
        upload_dir = "user_uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            
        # Save file with unique name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(x for x in uploaded_file.name if x.isalnum() or x in "._- ")
        save_path = os.path.join(upload_dir, f"{timestamp}_{safe_filename}")
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Create scenario name
        hub_map = {
            "HB_NORTH": "North Hub", "HB_SOUTH": "South Hub", "HB_WEST": "West Hub", "HB_HOUSTON": "Houston Hub"
        }
        friendly_hub = hub_map.get(custom_hub, custom_hub)
        
        # Create scenarios for all selected years
        added_count = 0
        for custom_year in custom_years:
            year_label = "All Years" if custom_year_selection == "All Years" else str(custom_year)
            name = f"{custom_year} Custom in {friendly_hub} ({int(custom_capacity)}MW)"
            if custom_no_curtailment:
                name += " [No Curtailment]"
            
            # Check for duplicates
            if not any(s['name'] == name for s in st.session_state.scenarios):
                new_scenario = {
                    "id": datetime.now().isoformat() + f"_{added_count}",
                    "name": name,
                    "year": custom_year,
                    "hub": custom_hub,
                    "tech": "Custom Upload",
                    "duration": "Full Year",
                    "month": None,
                    "capacity_mw": custom_capacity,
                    "vppa_price": custom_vppa_price,
                    "no_curtailment": custom_no_curtailment,
                    "custom_profile_path": save_path
                }
                st.session_state.scenarios.append(new_scenario)
                added_count += 1
        
        if added_count > 0:
            st.sidebar.success(f"✅ Added {added_count} scenario(s)")
        else:
            st.sidebar.warning("⚠️ Scenario(s) already exist")

else:
    # --- Solar/Wind Batch Form ---
    with st.sidebar.form("add_scenario_form"):
        st.subheader("Add Scenarios (Batch)")
        
        available_years = [2025, 2024, 2023, 2022, 2021, 2020]
        common_hubs = ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON", "HB_PAN"]
        
        s_techs = st.multiselect("Generation Source", ["Solar", "Wind"], default=["Solar"])
        if not s_techs:
            st.warning("Please select at least one technology.")

        st.markdown("*Select multiple years/hubs*")
        # Add "All Years" checkbox for batch mode
        select_all_years = st.checkbox("Select All Years")
        
        # Show multiselect with all years selected if checkbox is checked
        default_years = available_years if select_all_years else [2025]
        s_years = st.multiselect("Years", available_years, default=default_years)
        if not s_years:
            st.warning("Please select at least one year.")
        
        s_hubs = st.multiselect("Hubs", common_hubs, default=["HB_NORTH"])
        if not s_hubs:
            st.warning("Please select at least one hub.")
        
        # Duration Selection
        use_specific_month = st.checkbox("Filter by specific month")
        s_duration = "Specific Month" if use_specific_month else "Full Year"
        
        s_months = None
        if use_specific_month:
            all_months = [
                "January", "February", "March", "April", "May", "June", 
                "July", "August", "September", "October", "November", "December"
            ]
            s_months = st.multiselect("Months", all_months, default=["January"])
            if not s_months:
                st.warning("Please select at least one month.")
        
        s_capacity = st.number_input("Capacity (MW)", value=80.0, step=10.0)
        s_vppa_price = st.number_input("VPPA Price ($/MWh)", value=50.0, step=1.0)
        
        # Curtailment Option
        s_no_curtailment = st.checkbox("Remove $0 floor (No Curtailment)")

        # TMY Override
        s_force_tmy = st.checkbox("Force TMY Data (Override Actuals)", value=False, help="Use typical weather data even for 2024.")
        
        
        st.markdown("---")
        
        # Two buttons: Add (append) vs Clear & Run (reset)
        # Three buttons: Add (append), Clear & Run (replace), Reset All (clear)
        col1, col2, col3 = st.columns([1.2, 1.2, 1])
        with col1:
            add_button = st.form_submit_button("➕ Add", type="primary", use_container_width=True)
        with col2:
            clear_run_button = st.form_submit_button("🔄 Clear & Run", type="secondary", use_container_width=True)
        with col3:
            reset_all_button = st.form_submit_button("🗑️ Reset", type="secondary", use_container_width=True)
        
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
                                
                                if s_force_tmy:
                                    name += " [TMY]"
                                    
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
                                        "force_tmy": s_force_tmy,
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
                                
                                if s_force_tmy:
                                    name += " [TMY]"
                                    
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
                                    "force_tmy": s_force_tmy,
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
            st.session_state.scenarios = []
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

if not st.session_state.scenarios:
    st.info("👈 Please add scenarios using the sidebar to begin comparison.")
    st.stop()

# Calculate Results
results = []
progress_bar = st.progress(0)

for i, scenario in enumerate(st.session_state.scenarios):
    # Fetch Data
    df_rtm = get_ercot_data(scenario['year'])
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
        fig_settle.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_settle.update_yaxes(title="Total Settlement ($)")
        fig_settle.update_xaxes(title="Scenario")
        fig_settle.update_layout(showlegend=True, legend_title_text="Scenario")
        
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
            marker_line_width=0
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
            margin=dict(t=80, b=60, l=60, r=20)
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
        df_rtm = get_ercot_data(selected_scenario_config['year'])
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

