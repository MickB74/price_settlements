from pathlib import Path
from datetime import datetime, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


WORKBOOK_PATH = Path(__file__).resolve().parents[1] / "VPPA_8760s.xlsx"


def _extract_default_year_from_summary(raw_df: pd.DataFrame) -> int:
    for cell in raw_df.iloc[0].tolist():
        if isinstance(cell, (datetime, pd.Timestamp)):
            return int(cell.year)
    return int(datetime.now().year)


def _to_hour_number(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (datetime, pd.Timestamp)):
        return float(value.hour)
    if isinstance(value, time):
        return float(value.hour)
    if isinstance(value, str):
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return float(parsed.hour)
        return np.nan
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        if 0.0 <= float(value) < 1.0:
            return float(round(float(value) * 24.0))
        return float(value)
    return np.nan


@st.cache_data(show_spinner=False)
def load_vppa_summary_profiles(path_str: str, year: int):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")

    raw = pd.read_excel(path, sheet_name="Summary", header=None)
    default_year = _extract_default_year_from_summary(raw)

    header_row = None
    for idx, row in raw.iterrows():
        vals = [str(v).strip().lower() if pd.notna(v) else "" for v in row.tolist()]
        if len(vals) >= 3 and vals[0] == "month" and vals[1] == "day" and vals[2] == "hour":
            header_row = idx
            break
    if header_row is None:
        raise ValueError("Could not find Month/Day/Hour header row in Summary sheet.")

    headers = [str(v).strip() if pd.notna(v) else f"col_{i}" for i, v in enumerate(raw.iloc[header_row].tolist())]
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = headers
    data = data.dropna(subset=["Month", "Day", "Hour"], how="any")

    data["Month"] = pd.to_numeric(data["Month"], errors="coerce")
    data["Day"] = pd.to_numeric(data["Day"], errors="coerce")
    data["Hour"] = data["Hour"].apply(_to_hour_number)
    data = data.dropna(subset=["Month", "Day", "Hour"])

    data["Month"] = data["Month"].astype(int)
    data["Day"] = data["Day"].astype(int)
    data["Hour"] = data["Hour"].astype(int).clip(lower=0, upper=23)

    timestamps = pd.to_datetime(
        {
            "year": int(year),
            "month": data["Month"],
            "day": data["Day"],
            "hour": data["Hour"],
        },
        errors="coerce",
    )
    data["Time_Central"] = timestamps
    data = data.dropna(subset=["Time_Central"]).copy()

    profile_cols = [c for c in data.columns if c not in {"Month", "Day", "Hour", "Time_Central"}]
    for col in profile_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    hourly = (
        data[["Time_Central"] + profile_cols]
        .sort_values("Time_Central")
        .drop_duplicates(subset=["Time_Central"], keep="first")
        .set_index("Time_Central")
    )
    return hourly, profile_cols, default_year


def _extract_model_hourly_series(model_df: pd.DataFrame) -> pd.Series:
    if "Time_Central" in model_df.columns:
        ts = pd.to_datetime(model_df["Time_Central"], errors="coerce")
    else:
        ts = pd.to_datetime(model_df.index, errors="coerce")
    mask_valid = ts.notna()
    if not mask_valid.any():
        return pd.Series(dtype=float)

    working = model_df.loc[mask_valid].copy()
    ts = ts.loc[mask_valid]
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("US/Central", ambiguous=False, nonexistent="shift_forward")
    else:
        ts = ts.dt.tz_convert("US/Central")
    working["Time_Central"] = ts.dt.tz_localize(None)
    working = working.dropna(subset=["Time_Central"])
    if working.empty:
        return pd.Series(dtype=float)

    if "Gen_Energy_MWh" in working.columns:
        energy = pd.to_numeric(working["Gen_Energy_MWh"], errors="coerce").fillna(0.0)
    elif "Gen_MW" in working.columns:
        energy = pd.to_numeric(working["Gen_MW"], errors="coerce").fillna(0.0)
        diffs = working["Time_Central"].sort_values().diff().dropna()
        step_hours = 0.25
        if not diffs.empty:
            step_hours = max(1.0 / 60.0, float(diffs.median().total_seconds() / 3600.0))
        energy = energy * step_hours
    else:
        return pd.Series(dtype=float)

    hourly = (
        pd.DataFrame({"Time_Central": working["Time_Central"].dt.floor("h"), "Model_MWh": energy})
        .groupby("Time_Central", as_index=True)["Model_MWh"]
        .sum()
        .sort_index()
    )
    return hourly


def _build_metrics(model_s: pd.Series, profile_s: pd.Series) -> dict:
    delta = profile_s - model_s
    return {
        "Hours Compared": int(len(delta)),
        "Model Total (MWh)": float(model_s.sum()),
        "Profile Total (MWh)": float(profile_s.sum()),
        "Difference (MWh)": float(delta.sum()),
        "Difference (%)": float((delta.sum() / model_s.sum()) * 100.0) if model_s.sum() != 0 else np.nan,
        "MAE (MWh/h)": float(delta.abs().mean()),
        "RMSE (MWh/h)": float(np.sqrt(np.mean(np.square(delta)))),
        "Correlation": float(model_s.corr(profile_s)) if len(delta) > 1 else np.nan,
    }


def render():
    st.header("VPPA 8760 Comparison")
    st.caption("Load profiles from VPPA_8760s.xlsx and compare selected profiles against your latest Bill Validation model run.")

    if "val_preview_results" not in st.session_state:
        st.info("Run a model in Bill Validation first, then come back here to compare against the VPPA 8760 profiles.")
        return

    if not WORKBOOK_PATH.exists():
        st.error(f"Could not find workbook: {WORKBOOK_PATH}")
        return

    model_sources = list(st.session_state["val_preview_results"].keys())
    if not model_sources:
        st.warning("No model sources are available in session state.")
        return

    first_source = model_sources[0]
    default_year = int(datetime.now().year)
    sample_df = st.session_state["val_preview_results"][first_source]
    if "Time_Central" in sample_df.columns:
        sample_time = pd.to_datetime(sample_df["Time_Central"], errors="coerce").dropna()
        if not sample_time.empty:
            default_year = int(sample_time.iloc[0].year)

    raw_summary = pd.read_excel(WORKBOOK_PATH, sheet_name="Summary", header=None, nrows=1)
    workbook_default_year = _extract_default_year_from_summary(raw_summary)

    c1, c2 = st.columns([1, 1])
    with c1:
        model_name = st.selectbox("Model Source", model_sources, index=0, key="vppa_compare_model_source")
    with c2:
        selected_year = st.number_input(
            "Profile Year",
            min_value=2010,
            max_value=2100,
            value=int(default_year or workbook_default_year),
            step=1,
            key="vppa_compare_profile_year",
            help="The VPPA workbook has Month/Day/Hour only, so choose the year to align with model timestamps.",
        )

    try:
        profile_df, profile_cols, wb_year = load_vppa_summary_profiles(str(WORKBOOK_PATH), int(selected_year))
    except Exception as e:
        st.error(f"Could not load VPPA profiles: {e}")
        return

    st.caption(
        f"Loaded `{len(profile_df):,}` hourly rows from `{WORKBOOK_PATH.name}` (Summary sheet). "
        f"Workbook default year appears to be {wb_year}."
    )

    selected_profiles = st.multiselect(
        "Select VPPA profiles to compare",
        options=profile_cols,
        default=profile_cols[: min(2, len(profile_cols))],
        key="vppa_compare_selected_profiles",
    )
    include_total = st.checkbox(
        "Include total of selected profiles",
        value=True,
        key="vppa_compare_include_total",
    )

    if not selected_profiles:
        st.warning("Select at least one VPPA profile.")
        return

    model_df = st.session_state["val_preview_results"][model_name]
    model_hourly = _extract_model_hourly_series(model_df)
    if model_hourly.empty:
        st.error("Selected model source does not include usable generation data.")
        return

    compare_table = pd.DataFrame({"Model_MWh": model_hourly})
    for prof in selected_profiles:
        compare_table[prof] = profile_df[prof].reindex(compare_table.index)
    if include_total:
        compare_table["Selected Total"] = compare_table[selected_profiles].sum(axis=1, min_count=1)
    compare_table = compare_table.dropna(subset=selected_profiles, how="all")

    if compare_table.empty:
        st.error("No overlapping timestamps between model data and selected VPPA profiles.")
        return

    metric_rows = []
    compare_targets = selected_profiles + (["Selected Total"] if include_total else [])
    for target in compare_targets:
        joined = compare_table[["Model_MWh", target]].dropna()
        if joined.empty:
            continue
        metrics = _build_metrics(joined["Model_MWh"], joined[target])
        metrics["Profile"] = target
        metric_rows.append(metrics)

    if not metric_rows:
        st.error("No overlapping timestamps between model data and selected VPPA profiles.")
        return

    if metric_rows:
        mdf = pd.DataFrame(metric_rows)[
            [
                "Profile",
                "Hours Compared",
                "Model Total (MWh)",
                "Profile Total (MWh)",
                "Difference (MWh)",
                "Difference (%)",
                "MAE (MWh/h)",
                "RMSE (MWh/h)",
                "Correlation",
            ]
        ]
        st.subheader("Summary Metrics")
        st.dataframe(
            mdf.style.format(
                {
                    "Model Total (MWh)": "{:,.0f}",
                    "Profile Total (MWh)": "{:,.0f}",
                    "Difference (MWh)": "{:+,.0f}",
                    "Difference (%)": "{:+.2f}%",
                    "MAE (MWh/h)": "{:,.2f}",
                    "RMSE (MWh/h)": "{:,.2f}",
                    "Correlation": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    monthly = compare_table.copy()
    monthly["Month"] = monthly.index.to_period("M").astype(str)
    agg_cols = ["Model_MWh"] + compare_targets
    monthly_agg = monthly.groupby("Month", as_index=False)[agg_cols].sum()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Model", x=monthly_agg["Month"], y=monthly_agg["Model_MWh"]))
    for target in compare_targets:
        fig.add_trace(go.Bar(name=target, x=monthly_agg["Month"], y=monthly_agg[target]))
    fig.update_layout(
        barmode="group",
        title="Monthly Energy Comparison (MWh)",
        xaxis_title="Month",
        yaxis_title="MWh",
        legend_title="Series",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View hourly comparison data"):
        st.dataframe(compare_table, use_container_width=True)
