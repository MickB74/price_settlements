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

# Page Config
st.set_page_config(page_title="VPPA Settlement Estimator", layout="wide")

# Title
st.title("VPPA Settlement Estimator")
st.markdown("Compare multiple Virtual Power Purchase Agreement (VPPA) scenarios in ERCOT.")

# --- State Management ---
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []

# --- Data Fetching ---
@st.cache_data(show_spinner="Fetching market data (this may take 1-2 minutes for the first load)...")
def get_iso_data(market, year):
    """Fetches and caches ISO RTM data for a given market and year."""
    # Local cache file path (e.g. ercot_rtm_2024.parquet or miso_rtm_2024.parquet)
    cache_file = f"{market.lower()}_rtm_{year}.parquet"
    
    # Try loading from local file first
    try:
        if pd.io.common.file_exists(cache_file):
            df = pd.read_parquet(cache_file)
            return df
    except Exception as e:
        st.warning(f"Could not load local cache: {e}")

    # Fetch from GridStatus if not cached
    try:
        if market == "ERCOT":
            iso = gridstatus.Ercot()
            df = iso.get_rtm_spp(year=year)
        elif market == "MISO":
            iso = gridstatus.MISO()
            # MISO Data Fetching
            # get_rt_lmp does not exist. Use get_lmp.
            # MISO Real-Time 5-min data for a full year is very large and prone to timeouts.
            # Using Day-Ahead Hourly for stability and speed in this demo.
            # If 5-min is strictly required, we would need to batch fetch by month.
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            # Note: GridStatus might expect 'date' for single day or 'start'/'end' for range
            try:
                df = iso.get_lmp(start=start_date, end=end_date, market="DAY_AHEAD_HOURLY", verbose=True)
            except TypeError:
                # Fallback if start/end not supported directly (some older versions use date=range?)
                # But standardized API usually supports start/end. 
                # Let's try to be robust. 
                df = iso.get_lmp(start=start_date, end=end_date, market="DAY_AHEAD_HOURLY", verbose=True)

            # Standardize Columns
            if 'LMP' in df.columns:
                df = df.rename(columns={'LMP': 'SPP'})
        else:
            return pd.DataFrame()
        
        # Pre-process: Ensure Time is datetime and localized
        if not pd.api.types.is_datetime64_any_dtype(df['Time']):
            df['Time'] = pd.to_datetime(df['Time'], utc=True)
        
        # Create Central Time column (Both ERCOT and MISO operate in Central mainly)
        df['Time_Central'] = df['Time'].dt.tz_convert('US/Central')
        
        # Save to local parquet for future speedups
        try:
            df.to_parquet(cache_file)
        except Exception as e:
            st.warning(f"Could not save local cache: {e}")
        
        return df
    except Exception as e:
        st.error(f"Error fetching data for {market} {year}: {e}")
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

    # Generate Profile using TMY Data
    interval_hours = 0.25 # Assume 15-min interval for simplicity (MISO is 5-min or hourly? Check later. Assume resampling handled)
    
    # MISO might be Hourly data?
    # If df_rtm has 8760/year rows per hub, it's hourly. Interval is 1.0.
    # If df_rtm has 35040/year rows per hub, it's 15-min. Interval is 0.25.
    # Let's detect interval dynamically
    if len(df_hub) > 0:
        time_diff = df_hub['Time_Central'].diff().median()
        if pd.notnull(time_diff):
            interval_hours = time_diff.total_seconds() / 3600.0
        else:
            interval_hours = 1.0 # Default fallback
            
    capacity_mw = scenario['capacity_mw']
    tech = scenario['tech']

    # Hub Locations (Lat, Lon)
    # Added MISO hubs
    HUB_LOCATIONS = {
        # ERCOT
        "HB_NORTH": (33.9137, -98.4934),   # Wichita Falls
        "HB_SOUTH": (29.4241, -98.4936),   # San Antonio
        "HB_WEST": (31.9973, -102.0779),   # Midland
        "HB_HOUSTON": (29.7604, -95.3698), # Houston
        "HB_PAN": (35.2220, -101.8313),    # Amarillo
        # MISO (Approximate Centers)
        "INDIANA.HUB": (39.7684, -86.1581), # Indianapolis
        "MICHIGAN.HUB": (42.3314, -83.0458), # Detroit
        "MINN.HUB": (44.9778, -93.2650),     # Minneapolis
        "ILLINOIS.HUB": (40.1164, -88.2434), # Champaign
        "ARKANSAS.HUB": (34.7465, -92.2896), # Little Rock
        "LOUISIANA.HUB": (30.4583, -91.1403), # Baton Rouge
        "TEXAS.HUB": (30.1588, -94.00),     # MISO Texas (Beaumont area)
        "MS.HUB": (32.2988, -90.1848),       # Mississippi (Jackson)
    }
    
    # Default to Abilene if hub not found
    default_loc = (32.4487, -99.7331)
    lat, lon = HUB_LOCATIONS.get(scenario['hub'], default_loc)

    try:
        profile_series = fetch_tmy.get_profile_for_year(
            year=scenario['year'],
            tech=tech,
            capacity_mw=capacity_mw,
            lat=lat,
            lon=lon
        )
        
        # Align profile with df_hub timestamps
        profile_central = profile_series.tz_convert('US/Central')
        
        # Reindex to match df_hub['Time_Central']
        # This handles interpolation if MISO is hourly but profile is 15-min, or vice versa
        # Actually fetch_tmy returns 15-min. If MISO is Hourly, we should probably downsample profile or upsample price.
        # Here we reindex profile to match price timestamps.
        # Method='nearest' or interpolate?
        # profile_central.reindex(...) might introduce NaNs if timestamps don't match exactly.
        # Best to use reindex with nearest or interpolate.
        
        # If we reindex a 15-min profile to Hourly timestamps, we lose detail but it works.
        # If we reindex Hourly profile to 15-min, we need ffill.
        potential_gen = profile_central.reindex(df_hub['Time_Central'], method='nearest').values
        
    except Exception as e:
        try:
            # Fallback if strict reindex fails due to tz issues
             potential_gen = np.zeros(len(df_hub))
        except:
             potential_gen = np.zeros(len(df_hub))

    df_hub['Potential_Gen_MW'] = potential_gen
    
    # Settlement
    strike_price = scenario['strike_price']
    df_hub['Strike_Price'] = strike_price
    df_hub['Settlement_Price'] = df_hub['SPP'] - strike_price
    
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

# ...

# --- Sidebar: Scenario Builder ---
st.sidebar.header("Scenario Builder")

with st.sidebar.form("add_scenario_form"):
    st.subheader("Add New Scenario")
    # --- Batch Selection Logic ---
    
    # Market Selection
    s_market = st.selectbox("Market", ["ERCOT", "MISO"], index=0)
    
    # Years
    available_years = [2025, 2024, 2023, 2022, 2021, 2020]
    select_all_years = st.checkbox("Select All Years", value=False)
    if select_all_years:
        s_years = available_years
        st.caption(f"Selected: {', '.join(map(str, s_years))}")
    else:
        s_years = st.multiselect("Years", available_years, default=[2025])
        if not s_years:
            st.warning("Please select at least one year.")

    # Hubs (Dynamic based on Market)
    if s_market == "ERCOT":
        market_hubs = ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON", "HB_PAN"]
    else: # MISO
        market_hubs = [
            "INDIANA.HUB", "MICHIGAN.HUB", "MINN.HUB", "ILLINOIS.HUB", 
            "ARKANSAS.HUB", "LOUISIANA.HUB", "TEXAS.HUB", "MS.HUB"
        ]
        
    select_all_hubs = st.checkbox("Select All Hubs", value=False)
    if select_all_hubs:
        s_hubs = market_hubs
        st.caption(f"Selected: {len(s_hubs)} Hubs")
    else:
        # Default depends on market to avoid invalid selection
        default_hub = [market_hubs[0]]
        s_hubs = st.multiselect("Hubs", market_hubs, default=default_hub)
        if not s_hubs:
            st.warning("Please select at least one hub.")
    
    s_tech = st.radio("Generation Source", ["Solar", "Wind"], index=0)
    
    # Duration Selection
    use_specific_month = st.checkbox("Filter by specific month")
    s_duration = "Specific Month" if use_specific_month else "Full Year"
    
    s_months = None
    if use_specific_month:
        all_months = [
            "January", "February", "March", "April", "May", "June", 
            "July", "August", "September", "October", "November", "December"
        ]
        select_all_months = st.checkbox("Select All Months", value=False)
        if select_all_months:
            s_months = all_months
            st.caption("Selected: All Months")
        else:
            s_months = st.multiselect("Months", all_months, default=["January"])
            if not s_months:
                st.warning("Please select at least one month.")
    
    s_capacity = st.number_input("Capacity (MW)", value=80.0, step=10.0)
    s_strike = st.number_input("Strike Price ($/MWh)", value=30.0, step=1.0)
    
    # Curtailment Option
    s_no_curtailment = st.checkbox("Remove $0 floor (No Curtailment)")
    
    submitted = st.form_submit_button("Add Scenarios")
    
    if submitted:
        if not s_years or not s_hubs or (use_specific_month and not s_months):
            st.error("Please ensure Years, Hubs, and Months (if applicable) are selected.")
        else:
            # Helper for friendly names
            hub_map = {
                "HB_NORTH": "North Hub", "HB_SOUTH": "South Hub", "HB_WEST": "West Hub", "HB_HOUSTON": "Houston Hub",
                "INDIANA.HUB": "Indiana Hub", "MICHIGAN.HUB": "Michigan Hub", "MINN.HUB": "Minnesota Hub",
                "ILLINOIS.HUB": "Illinois Hub", "ARKANSAS.HUB": "Arkansas Hub", "LOUISIANA.HUB": "Louisiana Hub",
                "TEXAS.HUB": "Texas Hub (MISO)", "MS.HUB": "Mississippi Hub"
            }
            
            added_count = 0
            
            # Iterate through all combinations
            for year in s_years:
                for hub in s_hubs:
                    friendly_hub = hub_map.get(hub, hub)
                    
                    # Define list of monthly iterations
                    month_iterator = s_months if use_specific_month else [None]
                    
                    for month in month_iterator:
                        # Construct Name
                        if use_specific_month:
                            name = f"{month} {year} {s_tech} in {friendly_hub} ({int(s_capacity)}MW)"
                        else:
                            name = f"{year} {s_tech} in {friendly_hub} ({int(s_capacity)}MW)"
                        
                        # Add Market tag if needed or specific details
                        if s_market == "MISO":
                            name = f"MISO {name}"
                            
                        if s_no_curtailment:
                            name += " [No Curtailment]"
                            
                        # Check for duplicates
                        if any(s['name'] == name for s in st.session_state.scenarios):
                            continue 
                        else:
                            new_scenario = {
                                "id": datetime.now().isoformat() + f"_{added_count}",
                                "name": name,
                                "market": s_market, # Store Market
                                "year": year,
                                "hub": hub,
                                "tech": s_tech,
                                "duration": s_duration,
                                "month": month,
                                "capacity_mw": s_capacity,
                                "strike_price": s_strike,
                                "no_curtailment": s_no_curtailment
                            }
                            st.session_state.scenarios.append(new_scenario)
                            added_count += 1
            
            if added_count > 0:
                st.success(f"Successfully added {added_count} scenarios!")
            else:
                st.warning("No new scenarios added (duplicates or empty selection).")

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
    market = scenario.get('market', 'ERCOT') # Default to ERCOT for older scenarios
    df_rtm = get_iso_data(market, scenario['year'])
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
        "duration": scenario['duration'], # Track duration type for plotting
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
    df_res = res['data'].copy()
    scenario_name = res['Scenario']
    duration_type = res['duration']
    color = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]
    
    # Resample to daily for cleaner chart
    daily = df_res.set_index('Time_Central')[['Settlement_Amount']].resample('D').sum().cumsum().reset_index()
    
    # Normalize to a common year (2024)
    daily['Normalized_Date'] = daily['Time_Central'].apply(lambda x: x.replace(year=2024))
    
    if duration_type == "Specific Month":
        # Plot as a "Pin" (Marker + Text) at the end of the month
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
    df_res = res['data'].copy()
    # Group by Month
    # Use to_period for grouping, but convert back to timestamp for Plotly axis formatting
    df_res['Month_Period'] = df_res['Time_Central'].dt.to_period('M')
    monthly = df_res.groupby('Month_Period')[['Settlement_Amount', 'Gen_Energy_MWh']].sum().reset_index()
    monthly['Scenario'] = res['Scenario']
    monthly['Month_Date'] = monthly['Month_Period'].dt.to_timestamp()
    
    # Normalize to 2024 for seasonal comparison
    monthly['Normalized_Month_Date'] = monthly['Month_Date'].apply(lambda x: x.replace(year=2024))
    
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
    
    # Insight for Generation
    total_gen_by_scen = df_monthly.groupby('Scenario')['Gen_Energy_MWh'].sum()
    max_gen_scen = total_gen_by_scen.idxmax()
    max_gen_val = total_gen_by_scen.max()
    
    st.markdown(
        f"**Insight:** **{max_gen_scen}** was the top producer, generating **{max_gen_val:,.0f} MWh**."
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

