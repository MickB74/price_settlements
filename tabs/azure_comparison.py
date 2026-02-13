import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime, date, timedelta
import fetch_tmy
import sced_fetcher

# --- Configuration ---
CACHE_DIR = "sced_cache"
EXCLUDED_DATES = [
    date(2024, 1, 15), date(2024, 1, 16), # Winter Storm Heather (curtailment/icing)
]

def _index_diagnostics(idx: pd.Index):
    """Lightweight diagnostics for timestamp index alignment checks."""
    if idx is None or len(idx) == 0:
        return {
            "rows": 0,
            "start": None,
            "end": None,
            "tz": None,
            "duplicates": 0,
        }
    return {
        "rows": int(len(idx)),
        "start": idx.min(),
        "end": idx.max(),
        "tz": str(getattr(idx, "tz", None)),
        "duplicates": int(idx.duplicated().sum()),
    }

def load_data(year):
    """
    Loads Actuals (TWA) and Modeled (Mixed Fleet) data for the given year.
    Returns a merged dataframe with 'Actual' and 'Predicted' columns.
    """
    
    # 1. Load Actuals (TWA)
    actual_file = os.path.join(CACHE_DIR, f"AZURE_SKY_WIND_AGG_{year}_full.parquet")
    st.write(f"Debug: Looking for actual file at {actual_file}") # DEBUG
    st.write(f"Debug: Exists? {os.path.exists(actual_file)}") # DEBUG
    
    if not os.path.exists(actual_file):
        # Try to fetch on the fly if missing (e.g. for current year)
        if year == datetime.now().year:
            st.toast(f"Fetching current year data for {year}...", icon="⏳")
            end_date = date.today()
            sced_fetcher.get_asset_period_data("AZURE_SKY_WIND_AGG", date(year, 1, 1), end_date)
            sced_fetcher.consolidate_year("AZURE_SKY_WIND_AGG", year)
        
        if not os.path.exists(actual_file):
            st.error(f"Cache file not found: {actual_file}")
            return pd.DataFrame(), {}
            
    df_actual = pd.read_parquet(actual_file)
    
    # Standardize Actuals
    # TWA logic outputs: Time, Actual_MW, MWh_interval, coverage
    df_actual = df_actual.rename(columns={"Actual_MW": "Actual"})
    df_actual = df_actual.set_index("Time").sort_index()
    
    # Convert to Central Time for display/alignment
    if df_actual.index.tz is None:
        df_actual.index = df_actual.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        df_actual.index = df_actual.index.tz_convert("US/Central")


    # 2. Load Modeled (Mixed Fleet)
    # Azure Sky Config
    LAT = 33.1534
    LON = -99.2847
    TURBINES = [
        {'type': 'NORDEX_N149', 'count': 65, 'capacity_mw': 4.5, 'hub_height_m': 105.0},
        {'type': 'VESTAS_V163', 'count': 7,  'capacity_mw': 3.45, 'hub_height_m': 82.0},
        {'type': 'GENERIC',     'count': 7,  'capacity_mw': 2.0,  'hub_height_m': 80.0},
    ]
    CAPACITY = sum(t['count'] * t['capacity_mw'] for t in TURBINES) # ~330.65 MW
    
    with st.spinner(f"Generating modeled profile for {year}..."):
        try:
            s_modeled = fetch_tmy.get_blended_profile_for_year(
                year=year,
                tech="Wind",
                turbines=TURBINES,
                lat=LAT,
                lon=LON
            )
        except Exception as e:
            st.error(f"Error generating modeled profile: {e}")
            return pd.DataFrame(), {}
        
    if s_modeled.empty:
        st.warning(f"No modeled data available for {year}.")
        return pd.DataFrame(), {}

    # Align Modeled
    s_modeled.name = "Predicted"
    # Modeled is usually timezone-naive (local time) or UTC? 
    # fetch_tmy returns UTC-indexed series usually. Let's verify.
    # If naive, assume UTC then convert.
    if s_modeled.index.tz is None:
        s_modeled.index = s_modeled.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        s_modeled.index = s_modeled.index.tz_convert("US/Central")
        
    # 3. Merge
    df_merged = pd.merge(
        df_actual[["Actual"]], 
        s_modeled, 
        left_index=True, 
        right_index=True, 
        how="inner"
    )

    actual_diag = _index_diagnostics(df_actual.index)
    modeled_diag = _index_diagnostics(s_modeled.index)
    merged_diag = _index_diagnostics(df_merged.index)
    overlap_count = int(df_actual.index.intersection(s_modeled.index).size)
    overlap_vs_actual = (overlap_count / actual_diag["rows"]) if actual_diag["rows"] else np.nan
    overlap_vs_modeled = (overlap_count / modeled_diag["rows"]) if modeled_diag["rows"] else np.nan
    debug_info = {
        "actual": actual_diag,
        "modeled": modeled_diag,
        "merged": merged_diag,
        "overlap_count": overlap_count,
        "overlap_vs_actual": overlap_vs_actual,
        "overlap_vs_modeled": overlap_vs_modeled,
    }
    
    # 4. Calculate Residuals
    df_merged["Residual"] = df_merged["Actual"] - df_merged["Predicted"]
    df_merged["Error_Pct"] = (df_merged["Residual"] / CAPACITY) * 100
    
    return df_merged, debug_info

def render():
    try:
        st.header("Azure Sky Wind: Performance Analysis")
        st.write("Debug: Render function started...") # DEBUG
        
        year = st.selectbox("Select Year", [2024, 2025], index=0)
        st.write(f"Debug: Selected year {year}") # DEBUG
        
        df, diag = load_data(year)
        st.write(f"Debug: Data loaded. Rows: {len(df)}") # DEBUG
        if diag:
            st.write(
                "Debug: Merge diagnostics -> "
                f"Actual rows={diag['actual']['rows']:,}, "
                f"Modeled rows={diag['modeled']['rows']:,}, "
                f"Merged rows={diag['merged']['rows']:,}, "
                f"Overlap={diag['overlap_count']:,} "
                f"({diag['overlap_vs_actual']:.1%} of actual, {diag['overlap_vs_modeled']:.1%} of modeled)"
            )
            st.write(
                "Debug: Time ranges -> "
                f"Actual [{diag['actual']['start']} .. {diag['actual']['end']}], "
                f"Modeled [{diag['modeled']['start']} .. {diag['modeled']['end']}], "
                f"Merged [{diag['merged']['start']} .. {diag['merged']['end']}]"
            )
            st.write(
                "Debug: Timestamp quality -> "
                f"Actual tz={diag['actual']['tz']}, dupes={diag['actual']['duplicates']}; "
                f"Modeled tz={diag['modeled']['tz']}, dupes={diag['modeled']['duplicates']}; "
                f"Merged dupes={diag['merged']['duplicates']}"
            )
        
        if df.empty:
            st.warning("No data found for selected year.")
            return

        # --- Metrics ---
        # Date/Time Filter
        min_date = df.index.min().date()
        max_date = df.index.max().date()
        
        with st.expander("📅 Filter Time Period", expanded=False):
            filter_mode = st.radio("Filter Mode", ["Date Range", "Specific Months"], horizontal=True)
            
            df_filtered = df.copy()
            
            if filter_mode == "Date Range":
                date_range = st.date_input(
                    "Select Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="azure_date_range"
                )
                if len(date_range) == 2:
                    start_d, end_d = date_range
                    mask = (df.index.date >= start_d) & (df.index.date <= end_d)
                    df_filtered = df.loc[mask]
                    period_label = f"{start_d} to {end_d}"
                else:
                    period_label = "Full Year"

            else: # Specific Months
                all_months = ["January", "February", "March", "April", "May", "June", 
                              "July", "August", "September", "October", "November", "December"]
                # Default to all available months in data
                available_months = sorted(list(set(df.index.strftime('%B'))), key=lambda x: datetime.strptime(x, '%B').month)
                
                selected_months = st.multiselect(
                    "Select Months", 
                    all_months, 
                    default=available_months,
                    key="azure_month_picker"
                )
                
                if selected_months:
                    # Filter by month name
                    mask = df.index.strftime('%B').isin(selected_months)
                    df_filtered = df.loc[mask]
                    period_label = ", ".join(selected_months[:3]) + ("..." if len(selected_months) > 3 else "")
                else:
                    st.warning("Please select at least one month.")
                    return

        # Maintenance Filter
        with st.expander("🛠️ Operational Filters", expanded=False):
            exclude_maintenance = st.checkbox(
                "Exclude Potential Maintenance / Forced Outages", 
                value=False,
                help="Excludes periods where Modeled > 20 MW but Actual < 2 MW (likely downtime)."
            )
            
        # Apply Maintenance Filter
        if exclude_maintenance:
            # Identifies periods where we *should* be generating but aren't
            maintenance_mask = (df_filtered["Predicted"] > 20) & (df_filtered["Actual"] < 2)
            n_excluded = maintenance_mask.sum()
            hours_excluded = n_excluded * 0.25
            
            if n_excluded > 0:
                st.info(f"ℹ️ Excluded {n_excluded} intervals ({hours_excluded:.1f} hours) of potential downtime.")
                df_filtered = df_filtered[~maintenance_mask]
            else:
                st.info("No potential downtime events found in selected period.")

        if df_filtered.empty:
            st.warning("No data remains after filtering.")
            return

        st.write(f"Debug: Rows after filters: {len(df_filtered):,}") # DEBUG

        # --- Metrics ---
        # Calculate on FILTERED data
        correlation = df_filtered["Actual"].corr(df_filtered["Predicted"])
        rmse = np.sqrt(((df_filtered["Predicted"] - df_filtered["Actual"]) ** 2).mean())
        mbe = (df_filtered["Actual"] - df_filtered["Predicted"]).mean()
        
        # Energy Totals (Scale to GWh)
        # Note: If we exclude maintenance, "Actual" will naturally be higher relative to "Modeled" 
        # than before because we removed the zeros.
        total_energy_act = df_filtered["Actual"].sum() * 0.25 / 1000 # GWh
        total_energy_pred = df_filtered["Predicted"].sum() * 0.25 / 1000 # GWh
        delta_energy = total_energy_act - total_energy_pred
        
        st.markdown(f"### Performance Metrics ({period_label})")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        
        m1.metric("Actual Energy", f"{total_energy_act:,.1f} GWh")
        m2.metric("Modeled Energy", f"{total_energy_pred:,.1f} GWh")
        m3.metric("Net Delta", f"{delta_energy:,.1f} GWh", 
                  delta_color="normal" if abs(delta_energy) < 10 else "inverse")
        m4.metric("Correlation", f"{correlation:.3f}")
        m5.metric("RMSE", f"{rmse:.1f} MW")

        # --- Time Series ---
        st.subheader("Generation Timeline")
        st.caption("Zoom in to inspect specific events. 'Actual' uses 15-min Time-Weighted Average.")
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered["Predicted"], name="Modeled (Mixed Fleet)", line=dict(color='gray', width=1)))
        fig_ts.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered["Actual"], name="Actual (SCED TWA)", line=dict(color='#0068C9', width=1.5)))
        
        fig_ts.update_layout(
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            yaxis=dict(title="Generation (MW)"),
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right")
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # --- Residual Heatmap ---
        st.subheader("Error Heatmap (Actual - Predicted)")
        st.caption("Red = Over-performance (Actual > Predicted), Blue = Under-performance (Actual < Predicted)")
        
        # Need to re-create date/hour columns on filtered data
        df_filtered = df_filtered.copy()
        df_filtered["Hour"] = df_filtered.index.hour
        df_filtered["Date"] = df_filtered.index.date
        
        # Pivot for Heatmap: Index=Date, Columns=Hour, Values=Residual
        # We aggregate by mean in case of duplicate hourly entries (shouldn't happen with 15-min, but safe for heatmap)
        heatmap_data = df_filtered.groupby(["Date", "Hour"])["Residual"].mean().unstack()
        
        fig_heat = px.imshow(
            heatmap_data, 
            aspect="auto",
            color_continuous_scale="RdBu_r", # Red high, Blue low. 
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

        # --- Scatter Plot ---
        st.subheader("Correlation Scatter")
        col_scat, col_stats = st.columns([3, 1])
        
        with col_scat:
            fig_scat = px.scatter(
                df_filtered, x="Predicted", y="Actual", 
                trendline="ols", 
                opacity=0.3,
                color_discrete_sequence=["#0068C9"],
                labels={"Predicted": "Modeled MW", "Actual": "Actual MW"}
            )
            fig_scat.add_shape(type="line", x0=0, y0=0, x1=350, y1=350, line=dict(color="Red", width=1, dash="dash"))
            st.plotly_chart(fig_scat, use_container_width=True)
            
        with col_stats:
            st.markdown("##### Quick Stats")
            st.markdown(f"**Data Points:** {len(df_filtered):,}")
            st.markdown(f"**Mean Actual:** {df_filtered['Actual'].mean():.1f} MW")
            st.markdown(f"**Mean Modeled:** {df_filtered['Predicted'].mean():.1f} MW")
            st.markdown(f"**Max Actual:** {df_filtered['Actual'].max():.1f} MW")
    
    except Exception as e:
        st.error(f"CRITICAL ERROR in Render: {e}")
        import traceback
        st.code(traceback.format_exc())
