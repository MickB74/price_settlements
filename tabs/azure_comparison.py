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

def load_data(year):
    """
    Loads Actuals (TWA) and Modeled (Mixed Fleet) data for the given year.
    Returns a merged dataframe with 'Actual' and 'Predicted' columns.
    """
    
    # 1. Load Actuals (TWA)
    actual_file = os.path.join(CACHE_DIR, f"AZURE_SKY_WIND_AGG_{year}_full.parquet")
    if not os.path.exists(actual_file):
        # Try to fetch on the fly if missing (e.g. for current year)
        if year == datetime.now().year:
            st.toast(f"Fetching current year data for {year}...", icon="⏳")
            end_date = date.today()
            sced_fetcher.get_asset_period_data("AZURE_SKY_WIND_AGG", date(year, 1, 1), end_date)
            sced_fetcher.consolidate_year("AZURE_SKY_WIND_AGG", year)
        
        if not os.path.exists(actual_file):
            st.error(f"Cache file not found: {actual_file}")
            return pd.DataFrame()
            
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
            return pd.DataFrame()
        
    if s_modeled.empty:
        st.warning(f"No modeled data available for {year}.")
        return pd.DataFrame()

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
    
    # 4. Calculate Residuals
    df_merged["Residual"] = df_merged["Actual"] - df_merged["Predicted"]
    df_merged["Error_Pct"] = (df_merged["Residual"] / CAPACITY) * 100
    
    return df_merged

def render():
    st.header("Azure Sky Wind: Performance Analysis")
    
    year = st.selectbox("Select Year", [2024, 2025], index=0)
    
    df = load_data(year)
    
    if df.empty:
        st.warning("No data found for selected year.")
        return

    # --- Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    
    correlation = df["Actual"].corr(df["Predicted"])
    rmse = np.sqrt(((df["Predicted"] - df["Actual"]) ** 2).mean())
    mbe = (df["Actual"] - df["Predicted"]).mean()
    total_energy_act = df["Actual"].sum() * 0.25 / 1000 # GWh
    total_energy_pred = df["Predicted"].sum() * 0.25 / 1000 # GWh
    
    col1.metric("Correlation (R)", f"{correlation:.3f}")
    col2.metric("RMSE (MW)", f"{rmse:.1f}")
    col3.metric("Bias (MBE)", f"{mbe:.1f} MW")
    col4.metric("Energy Delta", f"{total_energy_act - total_energy_pred:.1f} GWh", 
                delta_color="normal" if abs(total_energy_act - total_energy_pred) < 10 else "inverse")

    # --- Time Series ---
    st.subheader("Generation Timeline")
    st.caption("Zoom in to inspect specific events. 'Actual' uses 15-min Time-Weighted Average.")
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=df.index, y=df["Predicted"], name="Modeled (Mixed Fleet)", line=dict(color='gray', width=1)))
    fig_ts.add_trace(go.Scatter(x=df.index, y=df["Actual"], name="Actual (SCED TWA)", line=dict(color='#0068C9', width=1.5)))
    
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
    
    df["Hour"] = df.index.hour
    df["Date"] = df.index.date
    
    # Pivot for Heatmap: Index=Date, Columns=Hour, Values=Residual
    # We aggregate by mean in case of duplicate hourly entries (shouldn't happen with 15-min, but safe for heatmap)
    heatmap_data = df.groupby(["Date", "Hour"])["Residual"].mean().unstack()
    
    fig_heat = px.imshow(
        heatmap_data, 
        aspect="auto",
        color_continuous_scale="RdBu_r", # Red high, Blue low. 
        # But wait, Residual = Act - Pred. 
        # Positive (Red) = Act > Pred (Good/Over). 
        # Negative (Blue) = Act < Pred (Bad/Under). 
        # Usually Red is 'Bad'. Let's flip or use different scale.
        # Let's use Balance (Red-Blue). 
        # Maybe use specific modeled errors?
    )
    fig_heat.update_layout(height=400)
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- Scatter Plot ---
    st.subheader("Correlation Scatter")
    col_scat, col_stats = st.columns([3, 1])
    
    with col_scat:
        fig_scat = px.scatter(
            df, x="Predicted", y="Actual", 
            trendline="ols", 
            opacity=0.3,
            color_discrete_sequence=["#0068C9"],
            labels={"Predicted": "Modeled MW", "Actual": "Actual MW"}
        )
        fig_scat.add_shape(type="line", x0=0, y0=0, x1=350, y1=350, line=dict(color="Red", width=1, dash="dash"))
        st.plotly_chart(fig_scat, use_container_width=True)
        
    with col_stats:
        st.markdown("##### Quick Stats")
        st.markdown(f"**Data Points:** {len(df):,}")
        st.markdown(f"**Mean Actual:** {df['Actual'].mean():.1f} MW")
        st.markdown(f"**Mean Modeled:** {df['Predicted'].mean():.1f} MW")
        st.markdown(f"**Max Actual:** {df['Actual'].max():.1f} MW")
