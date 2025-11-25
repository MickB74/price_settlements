import streamlit as st
import pandas as pd
import numpy as np
import gridstatus
import patch_gridstatus # Apply monkey patch
import plotly.express as px
from datetime import datetime
import zipfile
import io

# Page Config
st.set_page_config(page_title="VPPA Settlement Estimator", layout="wide")

# Title
st.title("VPPA Settlement Estimator")
st.markdown("Compare multiple Virtual Power Purchase Agreement (VPPA) scenarios in ERCOT.")

# --- State Management ---
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []

# --- Data Fetching ---
@st.cache_data
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
        empty_df = pd.DataFrame(columns=['Time_Central', 'Potential_Gen_MW', 'Strike_Price', 'Settlement_Price', 'Actual_Gen_MW', 
                                       'Gen_Energy_MWh', 'Curtailed_MWh', 'Settlement_Amount', 'Cumulative_Settlement', 'SPP'])
        # Ensure correct dtypes
        for col in empty_df.columns:
            if col == 'Time_Central':
                empty_df[col] = pd.to_datetime(empty_df[col])
            else:
                empty_df[col] = pd.to_numeric(empty_df[col])
        return empty_df

    # Generate Profile
    interval_hours = 0.25
    capacity_mw = scenario['capacity_mw']
    tech = scenario['tech']
    
    times = df_hub['Time_Central']
    hours = times.dt.hour + times.dt.minute / 60.0
    
    if tech == "Solar":
        day_of_years = times.dt.dayofyear
        day_length = 12 + 2 * np.sin(2 * np.pi * (day_of_years - 80) / 365)
        solar_noon = 13.5
        sunrise = solar_noon - (day_length / 2)
        sunset = solar_noon + (day_length / 2)
        
        is_daytime = (hours > sunrise) & (hours < sunset)
        sin_arg = np.pi * (hours - sunrise) / day_length
        potential_gen = np.where(is_daytime, capacity_mw * np.sin(sin_arg), 0.0)
        potential_gen = np.maximum(potential_gen, 0.0)
        
    elif tech == "Wind":
        base_signal = np.sin((hours + 4) * 2 * np.pi / 24)
        normalized_base = (base_signal + 1) / 2
        scaled_base = 0.2 + 0.6 * normalized_base
        noise = 0.15 * np.sin(hours * 10)
        profile = scaled_base + noise
        profile = np.clip(profile, 0, 1)
        potential_gen = profile * capacity_mw
        
    df_hub['Potential_Gen_MW'] = potential_gen
    
    # Settlement
    strike_price = scenario['strike_price']
    df_hub['Strike_Price'] = strike_price
    df_hub['Settlement_Price'] = df_hub['SPP'] - strike_price
    
    # Curtailment
    # Default to "Market Price < 0" logic unless disabled
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

# ...

# --- Sidebar: Scenario Builder ---
st.sidebar.header("Scenario Builder")

with st.sidebar.form("add_scenario_form"):
    st.subheader("Add New Scenario")
    s_year = st.selectbox("Year", [2025, 2024, 2023], index=1)
    
    # Get hubs (fetch one year to get list if needed, or hardcode common ones)
    # Hardcoding for speed/reliability without fetching data just for dropdown
    common_hubs = ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON"]
    s_hub = st.selectbox("Hub", common_hubs, index=0) # Default HB_NORTH
    
    s_tech = st.radio("Generation Source", ["Solar", "Wind"], index=0)
    
    # Duration Selection
    use_specific_month = st.checkbox("Filter by specific month")
    s_duration = "Specific Month" if use_specific_month else "Full Year"
    
    s_month = st.selectbox("Month", [
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ])
    
    if not use_specific_month:
        s_month = None
    
    s_capacity = st.number_input("Capacity (MW)", value=80.0, step=10.0)
    s_strike = st.number_input("Strike Price ($/MWh)", value=30.0, step=1.0)
    
    # Curtailment Option
    s_no_curtailment = st.checkbox("Remove $0 floor (No Curtailment)")
    
    submitted = st.form_submit_button("Add Scenario")
    
    if submitted:
        if len(st.session_state.scenarios) >= 10:
            st.error("Maximum of 10 scenarios allowed. Please remove one first.")
        else:
            # Helper for friendly names
            hub_map = {
                "HB_NORTH": "North Hub",
                "HB_SOUTH": "South Hub",
                "HB_WEST": "West Hub",
                "HB_HOUSTON": "Houston Hub"
            }
            friendly_hub = hub_map.get(s_hub, s_hub)
            
            if s_duration == "Specific Month":
                name = f"{s_month} {s_year} {s_tech} in {friendly_hub} ({int(s_capacity)}MW, Strike {int(s_strike)})"
            else:
                name = f"{s_year} {s_tech} in {friendly_hub} ({int(s_capacity)}MW, Strike {int(s_strike)})"
            
            if s_no_curtailment:
                name += " [No Curtailment]"
                
            new_scenario = {
                "id": datetime.now().isoformat(),
                "name": name,
                "year": s_year,
                "hub": s_hub,
                "tech": s_tech,
                "duration": s_duration,
                "month": s_month,
                "capacity_mw": s_capacity,
                "strike_price": s_strike,
                "no_curtailment": s_no_curtailment
            }
            st.session_state.scenarios.append(new_scenario)
            st.success(f"Added: {name}")

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
    
    if st.sidebar.button("Clear All Scenarios"):
        st.session_state.scenarios = []
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
    
    results.append({
        "Scenario": scenario['name'],
        "Net Settlement ($)": total_rev,
        "Total Gen (MWh)": total_gen,
        "Curtailed (MWh)": total_curt,
        "Capture Price ($/MWh)": capture_price,
        "Avg Hub Price ($/MWh)": avg_price,
        "data": df_res # Store full dataframe for plotting
    })
    progress_bar.progress((i + 1) / len(st.session_state.scenarios))

progress_bar.empty()

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
df_summary = pd.DataFrame(results).drop(columns=["data"])
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

# Cumulative Data
cum_data = []
for res in results:
    df_res = res['data'].copy()
    # Resample to daily for cleaner chart
    daily = df_res.set_index('Time_Central')[['Settlement_Amount']].resample('D').sum().cumsum().reset_index()
    daily['Scenario'] = res['Scenario']
    cum_data.append(daily)

if cum_data:
    df_cum = pd.concat(cum_data, ignore_index=True)
    
    st.subheader("Cumulative Settlement ($)")
    
    # Insight for Cumulative
    final_vals = df_cum.groupby('Scenario')['Settlement_Amount'].last()
    best_scen = final_vals.idxmax()
    best_val = final_vals.max()
    worst_scen = final_vals.idxmin()
    worst_val = final_vals.min()
    
    if len(final_vals) > 1:
        st.markdown(
            f"**Insight:** The **{best_scen}** scenario leads with a total settlement of "
            f"**\${best_val:,.0f}**, while **{worst_scen}** trails at **\${worst_val:,.0f}**."
        )
    else:
        st.markdown(
            f"**Insight:** The **{best_scen}** scenario has a total settlement of **\${best_val:,.0f}**."
        )

    fig_cum = px.line(
        df_cum, 
        x='Time_Central', 
        y='Settlement_Amount', 
        color='Scenario',
        title="Cumulative Settlement Over Time",
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig_cum.update_yaxes(tickprefix="$", title="Settlement Amount ($)")
    fig_cum.update_xaxes(title="Date", tickformat="%b %Y")
    st.plotly_chart(fig_cum, use_container_width=True)

# Monthly Data
monthly_data = []
for res in results:
    df_res = res['data'].copy()
    # Group by Month
    # Use to_period for grouping, but convert back to timestamp for Plotly axis formatting
    df_res['Month_Period'] = df_res['Time_Central'].dt.to_period('M')
    monthly = df_res.groupby('Month_Period')[['Settlement_Amount', 'Gen_Energy_MWh']].sum().reset_index()
    monthly['Scenario'] = res['Scenario']
    monthly['Month_Date'] = monthly['Month_Period'].dt.to_timestamp()
    monthly_data.append(monthly)

if monthly_data:
    df_monthly = pd.concat(monthly_data, ignore_index=True)
    
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
        x='Month_Date', 
        y='Settlement_Amount', 
        color='Scenario', 
        barmode='group',
        title="Monthly Net Settlement",
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig_settle.update_yaxes(tickprefix="$", title="Settlement Amount ($)")
    fig_settle.update_xaxes(
        title="Month", 
        tickformat="%b %Y", 
        dtick="M1" # Force monthly ticks
    )
    st.plotly_chart(fig_settle, use_container_width=True)
    
    # Chart 3: Monthly Generation
    st.subheader("Monthly Generation (MWh)")
    
    # Insight for Generation
    total_gen_by_scen = df_monthly.groupby('Scenario')['Gen_Energy_MWh'].sum()
    max_gen_scen = total_gen_by_scen.idxmax()
    max_gen_val = total_gen_by_scen.max()
    
    st.markdown(
        f"**Insight:** **{max_gen_scen}** was the top producer, generating **{max_gen_val:,.0f} MWh**."
    )

    fig_gen = px.bar(
        df_monthly, 
        x='Month_Date', 
        y='Gen_Energy_MWh', 
        color='Scenario', 
        barmode='group',
        title="Monthly Energy Generation",
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig_gen.update_yaxes(title="Generation (MWh)")
    fig_gen.update_xaxes(
        title="Month", 
        tickformat="%b %Y", 
        dtick="M1"
    )
    st.plotly_chart(fig_gen, use_container_width=True)

# Data Preview
with st.expander("View Raw Data"):
    if results:
        # Scenario Selection
        scenario_names = [res['Scenario'] for res in results]
        selected_scenario_name = st.selectbox("Select Scenario", scenario_names)
        
        # Find selected result
        selected_result = next(res for res in results if res['Scenario'] == selected_scenario_name)
        df_display = selected_result['data']
        
        st.markdown(f"**Showing data for: {selected_scenario_name}**")
        st.dataframe(df_display)
        
        # Download CSV
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{selected_scenario_name}.csv",
            mime="text/csv",
        )
        
        st.markdown("---")
        
        # Download All as ZIP
        if st.button("Prepare ZIP of All Scenarios"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for res in results:
                    # Convert to CSV
                    csv_data = res['data'].to_csv(index=False)
                    # Add to zip with scenario name as filename (sanitize name if needed)
                    safe_name = "".join([c for c in res['Scenario'] if c.isalnum() or c in (' ', '-', '_')]).strip()
                    zf.writestr(f"{safe_name}.csv", csv_data)
            
            st.download_button(
                label="Download All Scenarios (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="vppa_scenarios_data.zip",
                mime="application/zip"
            )

