from pathlib import Path
from datetime import datetime, time
from collections import Counter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


WORKBOOK_PATH = Path(__file__).resolve().parents[1] / "VPPA_8760s.xlsx"
WORKBOOK_GLOB = "VPPA_8760*"
WORKBOOK_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}


def _list_local_workbooks():
    root = Path(__file__).resolve().parents[1]
    candidates = [
        p
        for p in sorted(root.glob(WORKBOOK_GLOB))
        if (not p.name.startswith("~$")) and p.is_file() and (p.suffix.lower() in WORKBOOK_EXTENSIONS)
    ]
    # Prefer the original workbook first if present.
    candidates.sort(key=lambda p: (0 if p.name == WORKBOOK_PATH.name else 1, p.name.lower()))
    return candidates


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


def _extract_offtake_map(raw: pd.DataFrame, header_row: int, headers: list[str]) -> dict:
    """Extract per-profile off-take MW from Summary rows above the data header."""
    offtake_row = None
    for idx in range(header_row):
        vals = [str(v).strip().lower() if pd.notna(v) else "" for v in raw.iloc[idx].tolist()]
        if any(("offtake" in v) or ("off-take" in v) for v in vals):
            offtake_row = idx
            break

    if offtake_row is None:
        return {}

    row_vals = raw.iloc[offtake_row].tolist()
    offtake_map = {}
    for i, col in enumerate(headers):
        if col in {"Month", "Day", "Hour", "Time_Central"}:
            continue
        val = row_vals[i] if i < len(row_vals) else np.nan
        num = pd.to_numeric(val, errors="coerce")
        if pd.notna(num):
            offtake_map[col] = float(num)
    return offtake_map


def _parse_summary_sheet(raw: pd.DataFrame, year: int):
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
    offtake_map = _extract_offtake_map(raw, int(header_row), headers)

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
    offtake_map = {col: offtake_map[col] for col in profile_cols if col in offtake_map}
    return hourly, profile_cols, default_year, offtake_map


@st.cache_data(show_spinner=False)
def load_vppa_summary_profiles_from_path(path_str: str, year: int, file_mtime_ns: int):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")
    raw = pd.read_excel(path, sheet_name="Summary", header=None)
    return _parse_summary_sheet(raw, year)


@st.cache_data(show_spinner=False)
def load_vppa_summary_profiles_from_bytes(file_bytes: bytes, year: int):
    raw = pd.read_excel(pd.io.common.BytesIO(file_bytes), sheet_name="Summary", header=None)
    return _parse_summary_sheet(raw, year)


@st.cache_data(show_spinner=False)
def load_local_profile_catalog(workbook_specs, year: int):
    """
    Load all local VPPA workbooks and return one combined profile table.
    workbook_specs: tuple of (path_str, file_mtime_ns)
    """
    project_entries = []
    default_years = []

    for path_str, file_mtime_ns in workbook_specs:
        hourly, profile_cols, default_year, wb_offtake_map = load_vppa_summary_profiles_from_path(
            path_str,
            int(year),
            int(file_mtime_ns),
        )
        default_years.append(int(default_year))
        source_name = Path(path_str).name
        for project in profile_cols:
            project_entries.append(
                (
                    project,
                    source_name,
                    pd.to_numeric(hourly[project], errors="coerce"),
                    wb_offtake_map.get(project),
                )
            )

    if not project_entries:
        return pd.DataFrame(), [], int(datetime.now().year), {}, {}

    counts = Counter(project for project, _, _, _ in project_entries)
    combined = pd.DataFrame()
    source_map = {}
    offtake_map = {}

    for project, source_name, series, offtake_mw in project_entries:
        label = project if counts[project] == 1 else f"{project} ({source_name})"
        if label in combined.columns:
            idx = 2
            while f"{label} [{idx}]" in combined.columns:
                idx += 1
            label = f"{label} [{idx}]"
        combined[label] = series
        source_map[label] = source_name
        if offtake_mw is not None and pd.notna(offtake_mw):
            offtake_map[label] = float(offtake_mw)

    combined = combined.sort_index()
    default_year = default_years[0] if default_years else int(datetime.now().year)
    return combined, list(combined.columns), int(default_year), source_map, offtake_map


def _normalize_model_timestamps(ts: pd.Series, time_alignment: str) -> pd.Series:
    # Normalize model timestamps into a consistent local timeline for VPPA alignment.
    # VPPA 8760 profiles are typically fixed standard time (no DST), so CST-no-DST
    # is offered as the default alignment mode.
    if getattr(ts.dt, "tz", None) is None:
        try:
            ts_local = ts.dt.tz_localize("US/Central", ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            ts_local = ts.dt.tz_localize("US/Central", ambiguous=False, nonexistent="shift_forward")
    else:
        ts_local = ts.dt.tz_convert("US/Central")

    if time_alignment == "CST_NO_DST":
        ts_local = ts_local.dt.tz_convert("Etc/GMT+6")

    return ts_local.dt.tz_localize(None)


def _extract_model_series(
    model_df: pd.DataFrame,
    interval_label: str = "Hourly",
    time_alignment: str = "CST_NO_DST",
    energy_basis: str = "SETTLED",
) -> pd.Series:
    if "Time_Central" in model_df.columns:
        ts = pd.to_datetime(model_df["Time_Central"], errors="coerce")
    else:
        ts = pd.to_datetime(model_df.index, errors="coerce")
    mask_valid = ts.notna()
    if not mask_valid.any():
        return pd.Series(dtype=float)

    working = model_df.loc[mask_valid].copy()
    ts = ts.loc[mask_valid]
    working["Time_Central"] = _normalize_model_timestamps(ts, time_alignment)
    working = working.dropna(subset=["Time_Central"])
    if working.empty:
        return pd.Series(dtype=float)

    if "Gen_Energy_MWh" in working.columns:
        settled = pd.to_numeric(working["Gen_Energy_MWh"], errors="coerce").fillna(0.0)
        if energy_basis == "POTENTIAL" and "Curtailed_MWh" in working.columns:
            curtailed = pd.to_numeric(working["Curtailed_MWh"], errors="coerce").fillna(0.0)
            energy = settled + curtailed
        else:
            energy = settled
    elif "Gen_MW" in working.columns:
        energy = pd.to_numeric(working["Gen_MW"], errors="coerce").fillna(0.0)
        diffs = working["Time_Central"].sort_values().diff().dropna()
        step_hours = 0.25
        if not diffs.empty:
            step_hours = max(1.0 / 60.0, float(diffs.median().total_seconds() / 3600.0))
        energy = energy * step_hours
    else:
        return pd.Series(dtype=float)

    native = (
        pd.DataFrame({"Time_Central": working["Time_Central"], "Model_MWh": energy})
        .groupby("Time_Central", as_index=True)["Model_MWh"]
        .sum()
        .sort_index()
    )

    if interval_label == "15-min":
        idx_diffs = native.index.to_series().sort_values().diff().dropna()
        if not idx_diffs.empty:
            native_step_hours = float(idx_diffs.median().total_seconds() / 3600.0)
            if native_step_hours >= 0.99:
                return _expand_hourly_series_to_15min(native)
        return native.groupby(native.index.floor("15min")).sum().sort_index()

    return native.groupby(native.index.floor("h")).sum().sort_index()


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


def _build_monthly_correlation(
    compare_table: pd.DataFrame,
    targets: list[str],
    count_label: str = "Hours Compared",
) -> pd.DataFrame:
    if compare_table.empty or not targets:
        return pd.DataFrame()

    working = compare_table.copy()
    working["Month"] = working.index.to_period("M").astype(str)
    rows = []

    for month, month_df in working.groupby("Month", sort=True):
        row = {"Month": month}
        min_hours = None
        for target in targets:
            pair = month_df[["Model_MWh", target]].dropna()
            hours = int(len(pair))
            if min_hours is None or hours < min_hours:
                min_hours = hours
            corr = pair["Model_MWh"].corr(pair[target]) if hours > 1 else np.nan
            row[target] = float(corr) if pd.notna(corr) else np.nan
        row[count_label] = int(min_hours or 0)
        rows.append(row)

    # Add a full-period row so total correlation is visible with monthly values.
    total_row = {"Month": "Total"}
    total_count = None
    for target in targets:
        pair = compare_table[["Model_MWh", target]].dropna()
        n_obs = int(len(pair))
        if total_count is None or n_obs < total_count:
            total_count = n_obs
        corr = pair["Model_MWh"].corr(pair[target]) if n_obs > 1 else np.nan
        total_row[target] = float(corr) if pd.notna(corr) else np.nan
    total_row[count_label] = int(total_count or 0)
    rows.append(total_row)

    if not rows:
        return pd.DataFrame()

    cols = ["Month", count_label] + targets
    return pd.DataFrame(rows)[cols]


def _expand_hourly_series_to_15min(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    expanded = series.reindex(series.index.repeat(4)).astype(float) / 4.0
    offsets = np.tile(np.array([0, 15, 30, 45], dtype=int), len(series))
    expanded.index = expanded.index + pd.to_timedelta(offsets, unit="m")
    return expanded.groupby(level=0).sum().sort_index()


def _expand_hourly_df_to_15min(profile_df: pd.DataFrame) -> pd.DataFrame:
    if profile_df.empty:
        return profile_df
    expanded = profile_df.reindex(profile_df.index.repeat(4)).copy() / 4.0
    offsets = np.tile(np.array([0, 15, 30, 45], dtype=int), len(profile_df))
    expanded.index = expanded.index + pd.to_timedelta(offsets, unit="m")
    return expanded.sort_index()


def _best_lag_correlation(model_s: pd.Series, profile_s: pd.Series, interval_label: str = "Hourly", max_hours: int = 24):
    step_minutes = 60 if interval_label == "Hourly" else 15
    max_steps = int((max_hours * 60) / step_minutes)
    best = {"Lag (hours)": 0.0, "Correlation": np.nan, "Points": 0}

    for step in range(-max_steps, max_steps + 1):
        shifted = profile_s.copy()
        shifted.index = shifted.index + pd.to_timedelta(step * step_minutes, unit="m")
        joined = pd.concat([model_s, shifted], axis=1, keys=["Model", "Profile"]).dropna()
        n = int(len(joined))
        corr = joined["Model"].corr(joined["Profile"]) if n > 1 else np.nan
        if pd.isna(corr):
            continue
        if pd.isna(best["Correlation"]) or corr > best["Correlation"]:
            best = {
                "Lag (hours)": float(step * step_minutes / 60.0),
                "Correlation": float(corr),
                "Points": n,
            }
    return best


def render():
    st.header("VPPA 8760 Comparison")
    st.caption("Load profiles from VPPA_8760s.xlsx and compare selected profiles against your latest Bill Validation model run.")

    if "val_preview_results" not in st.session_state:
        st.info("Run a model in Bill Validation first, then come back here to compare against the VPPA 8760 profiles.")
        return

    uploaded_wb = st.file_uploader(
        "VPPA Workbook (.xlsx/.xls/.xlsm)",
        type=["xlsx", "xls", "xlsm"],
        key="vppa_8760_workbook_upload",
        help="Upload a VPPA 8760 Excel workbook when the app environment does not have the file locally.",
    )
    local_workbooks = _list_local_workbooks()

    source_options = []
    if local_workbooks:
        source_options.append("Local file")
    source_options.append("Upload")
    source_mode = st.radio(
        "Workbook Source",
        options=source_options,
        index=0,
        horizontal=True,
        key="vppa_compare_source_mode",
        help="Choose whether to process a local VPPA workbook or the uploaded workbook.",
    )

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

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.2, 1.1])
    with c1:
        model_name = st.selectbox("Model Source", model_sources, index=0, key="vppa_compare_model_source")
    with c2:
        selected_year = st.number_input(
            "Profile Year",
            min_value=2010,
            max_value=2100,
            value=int(default_year),
            step=1,
            key="vppa_compare_profile_year",
            help="The VPPA workbook has Month/Day/Hour only, so choose the year to align with model timestamps.",
        )
    with c3:
        compare_interval = st.selectbox(
            "Comparison Interval",
            options=["Hourly", "15-min"],
            index=0,
            key="vppa_compare_interval",
            help="Choose whether to compare at hourly or 15-minute intervals.",
        )
    with c4:
        time_alignment_label = st.selectbox(
            "Time Alignment",
            options=["CST (No DST)", "US/Central (DST-aware)"],
            index=0,
            key="vppa_compare_time_alignment",
            help="VPPA 8760 profiles are usually fixed standard-time hours. Use CST (No DST) for best alignment.",
        )
    with c5:
        energy_basis_label = st.selectbox(
            "Model Energy Basis",
            options=["Settled (post-curtail)", "Potential (pre-curtail)"],
            index=1,
            key="vppa_compare_energy_basis",
            help="Potential adds curtailed MWh back before comparison.",
        )
    time_alignment = "CST_NO_DST" if time_alignment_label == "CST (No DST)" else "US_CENTRAL_DST"
    energy_basis = "POTENTIAL" if energy_basis_label == "Potential (pre-curtail)" else "SETTLED"

    if source_mode == "Local file":
        st.caption(f"Using all local VPPA workbooks found in the directory ({len(local_workbooks)} files).")
    elif uploaded_wb is None:
        st.caption("No workbook uploaded yet. Use the uploader above.")

    try:
        source_map = {}
        offtake_map = {}
        if source_mode == "Upload":
            if uploaded_wb is None:
                st.error("No workbook uploaded yet. Upload an Excel file above.")
                return
            profile_df, profile_cols, wb_year, offtake_map = load_vppa_summary_profiles_from_bytes(
                uploaded_wb.getvalue(),
                int(selected_year),
            )
            source_name = uploaded_wb.name
        elif local_workbooks:
            workbook_specs = tuple(
                (str(p), int(p.stat().st_mtime_ns))
                for p in local_workbooks
            )
            profile_df, profile_cols, wb_year, source_map, offtake_map = load_local_profile_catalog(
                workbook_specs,
                int(selected_year),
            )
            source_name = f"{len(local_workbooks)} local workbook(s)"
            if source_map:
                unique_sources = sorted(set(source_map.values()))
                st.caption("Loaded from: " + ", ".join(f"`{n}`" for n in unique_sources))
        else:
            st.error("Could not find a local workbook. Upload an Excel file above.")
            return
    except Exception as e:
        st.error(f"Could not load VPPA profiles: {e}")
        return

    st.caption(
        f"Loaded `{len(profile_df):,}` hourly rows from `{source_name}` (Summary sheet). "
        f"Workbook default year appears to be {wb_year}."
    )
    if compare_interval == "15-min":
        st.caption("15-min mode distributes each hourly VPPA value evenly across four quarter-hour intervals.")
    if time_alignment == "CST_NO_DST":
        st.caption("Using fixed CST (no DST) alignment for model timestamps.")
    else:
        st.caption("Using DST-aware US/Central alignment for model timestamps.")
    if energy_basis == "POTENTIAL":
        st.caption("Comparing against potential generation (curtailment added back).")

    profile_shift_hours = st.number_input(
        "VPPA Time Shift (hours)",
        min_value=-24,
        max_value=24,
        value=0,
        step=1,
        key="vppa_compare_profile_shift_hours",
        help="Shift VPPA profile timestamps to test hour alignment (positive moves VPPA later).",
    )

    def _format_profile_option(profile_name: str) -> str:
        mw = offtake_map.get(profile_name)
        if mw is None or pd.isna(mw):
            return profile_name
        mw_f = float(mw)
        mw_str = f"{mw_f:,.0f}" if mw_f.is_integer() else f"{mw_f:,.2f}".rstrip("0").rstrip(".")
        return f"{profile_name} ({mw_str} MW)"

    selected_profiles = st.multiselect(
        "Select VPPA profiles to compare",
        options=profile_cols,
        default=profile_cols[: min(2, len(profile_cols))],
        key="vppa_compare_selected_profiles",
        format_func=_format_profile_option,
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
    model_series = _extract_model_series(
        model_df,
        compare_interval,
        time_alignment=time_alignment,
        energy_basis=energy_basis,
    )
    if model_series.empty:
        st.error("Selected model source does not include usable generation data.")
        return

    profile_compare_df = profile_df if compare_interval == "Hourly" else _expand_hourly_df_to_15min(profile_df)
    if int(profile_shift_hours) != 0:
        profile_compare_df = profile_compare_df.copy()
        profile_compare_df.index = profile_compare_df.index + pd.to_timedelta(int(profile_shift_hours), unit="h")

    compare_table = pd.DataFrame({"Model_MWh": model_series})
    for prof in selected_profiles:
        compare_table[prof] = profile_compare_df[prof].reindex(compare_table.index)
    if include_total:
        compare_table["Selected Total"] = compare_table[selected_profiles].sum(axis=1, min_count=1)
    compare_table = compare_table.dropna(subset=selected_profiles, how="all")

    if compare_table.empty:
        st.error("No overlapping timestamps between model data and selected VPPA profiles.")
        return

    count_label = "Hours Compared" if compare_interval == "Hourly" else "Intervals Compared"
    error_label = "MAE (MWh/h)" if compare_interval == "Hourly" else "MAE (MWh/interval)"
    rmse_label = "RMSE (MWh/h)" if compare_interval == "Hourly" else "RMSE (MWh/interval)"
    metric_rows = []
    compare_targets = selected_profiles + (["Selected Total"] if include_total else [])
    for target in compare_targets:
        joined = compare_table[["Model_MWh", target]].dropna()
        if joined.empty:
            continue
        metrics = _build_metrics(joined["Model_MWh"], joined[target])
        metrics[count_label] = metrics.pop("Hours Compared")
        metrics[error_label] = metrics.pop("MAE (MWh/h)")
        metrics[rmse_label] = metrics.pop("RMSE (MWh/h)")
        metrics["Profile"] = target
        metric_rows.append(metrics)

    if not metric_rows:
        st.error("No overlapping timestamps between model data and selected VPPA profiles.")
        return

    if metric_rows:
        mdf = pd.DataFrame(metric_rows)[
            [
                "Profile",
                count_label,
                "Model Total (MWh)",
                "Profile Total (MWh)",
                "Difference (MWh)",
                "Difference (%)",
                error_label,
                rmse_label,
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
                    error_label: "{:,.2f}",
                    rmse_label: "{:,.2f}",
                    "Correlation": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    lag_rows = []
    for target in selected_profiles:
        joined = compare_table[["Model_MWh", target]].dropna()
        if joined.empty:
            continue
        best = _best_lag_correlation(
            joined["Model_MWh"],
            joined[target],
            interval_label=compare_interval,
            max_hours=24,
        )
        lag_rows.append(
            {
                "Profile": target,
                "Best Lag (hours)": best["Lag (hours)"],
                "Best Correlation": best["Correlation"],
                "Points": best["Points"],
            }
        )

    if lag_rows:
        with st.expander("Lag Diagnostic (best correlation within +/-24 hours)", expanded=False):
            st.dataframe(
                pd.DataFrame(lag_rows).style.format(
                    {
                        "Best Lag (hours)": "{:+.2f}",
                        "Best Correlation": "{:.3f}",
                        "Points": "{:,.0f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    monthly_corr_targets = selected_profiles + (["Selected Total"] if include_total else [])
    monthly_corr_df = _build_monthly_correlation(compare_table, monthly_corr_targets, count_label=count_label)
    if not monthly_corr_df.empty:
        corr_fmt = {c: "{:.3f}" for c in monthly_corr_targets if c in monthly_corr_df.columns}
        st.subheader("Monthly Correlation (Pearson R)")
        st.dataframe(
            monthly_corr_df.style.format(corr_fmt),
            use_container_width=True,
            hide_index=True,
        )

    monthly = compare_table.copy()
    monthly["Month"] = monthly.index.to_period("M").astype(str)
    # Keep chart focused on individual selected profiles; hide "Selected Total" from bars.
    chart_targets = selected_profiles
    agg_cols = ["Model_MWh"] + chart_targets
    monthly_agg = monthly.groupby("Month", as_index=False)[agg_cols].sum()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Model", x=monthly_agg["Month"], y=monthly_agg["Model_MWh"]))
    for target in chart_targets:
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
