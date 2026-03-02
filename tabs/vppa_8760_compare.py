from pathlib import Path
from datetime import datetime, time
from collections import Counter
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Bump this any time cached behaviour needs to be invalidated on deploy.
_CODE_VERSION = "2026-03-02-v4"


WORKBOOK_PATH = Path(__file__).resolve().parents[1] / "VPPA_8760s.xlsx"
WORKBOOK_GLOB = "VPPA_8760*"
WORKBOOK_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}
REGISTRY_PATH = Path(__file__).resolve().parents[1] / "ercot_assets.json"


def _load_registry() -> list[dict]:
    """Load ercot_assets.json and return a flat list of project dicts."""
    if not REGISTRY_PATH.exists():
        return []
    with open(REGISTRY_PATH) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return [v for v in raw.values() if isinstance(v, dict)]
    if isinstance(raw, list):
        return [v for v in raw if isinstance(v, dict)]
    return []


def _match_project_meta(project_name: str, registry: list[dict]) -> dict | None:
    """Fuzzy-match a project name to a registry entry (case-insensitive substring)."""
    needle = project_name.strip().lower()
    if not needle:
        return None
    # Exact match first
    for p in registry:
        candidate = (p.get("project_name") or p.get("name") or "").strip().lower()
        if candidate and candidate == needle:
            return p
    # Substring match — skip entries with no usable name
    for p in registry:
        candidate = (p.get("project_name") or p.get("name") or "").strip().lower()
        if not candidate:
            continue
        if needle in candidate or candidate in needle:
            return p
    return None


def _generate_weather_profile(
    project_name: str,
    tech: str,
    lat: float,
    lon: float,
    capacity_mw: float,
    year: int,
    turbine_model: str = "GENERIC",
    hub_height_m: float = 80.0,
    use_actual_weather: bool = True,
) -> pd.Series:
    """
    Generate an hourly generation profile for a single project.
    - use_actual_weather=True  →  Open-Meteo historical weather for the given year
    - use_actual_weather=False →  PVGIS TMY (typical-year average)
    - no curtailment, no SCED
    Returns an hourly Series indexed by tz-naive local (Central) timestamps.
    """
    import fetch_tmy  # local import to avoid circular

    series = fetch_tmy.get_profile_for_year(
        year=year,
        tech=tech,
        capacity_mw=capacity_mw,
        lat=lat,
        lon=lon,
        force_tmy=not use_actual_weather,
        turbine_type=turbine_model if tech == "Wind" else "GENERIC",
        hub_height=int(hub_height_m),
        efficiency=0.86,  # 14% losses
        apply_wind_calibration=False,  # pure model, no SCED bias correction
    )
    if series is None or series.empty:
        return pd.Series(dtype=float)

    # Normalise to tz-naive hourly Central time (same basis as VPPA 8760 sheets)
    if series.index.tz is not None:
        series = series.tz_convert("US/Central").tz_localize(None)

    # Resample to hourly if 15-min: use .mean() because values are
    # instantaneous MW — average MW over 1 hour equals MWh for that hour.
    if len(series) > 9000:
        series = series.resample("h").mean()

    # Hard-clip to nameplate AC capacity (model can exceed due to DC/AC ratio)
    series = series.clip(upper=capacity_mw)

    series = series[series.index.year == year]
    return series


def _list_local_workbooks():
    root = Path(__file__).resolve().parents[1]
    # Gather workbooks matching VPPA_8760* plus standalone Liberty.xlsx
    found = set()
    for pattern in (WORKBOOK_GLOB, "Liberty.xlsx"):
        for p in root.glob(pattern):
            if (not p.name.startswith("~$")) and p.is_file() and (p.suffix.lower() in WORKBOOK_EXTENSIONS):
                found.add(p)
    candidates = sorted(found, key=lambda p: p.name.lower())
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


def _parse_single_project_sheet(path: Path, year: int):
    """Parse a single-project workbook with Date/Hour/Gen MWh columns.

    Derives the profile column name from the filename (e.g. VPPA_8760s_Liberty.xlsx → Liberty).
    Returns (hourly_df, profile_cols, default_year, offtake_map) matching _parse_summary_sheet signature.
    """
    # Try common sheet names
    for sheet in ("Hourly Generation", "Sheet1", 0):
        try:
            raw = pd.read_excel(path, sheet_name=sheet, header=0)
            break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not find a usable sheet in {path.name}")

    # Normalise column names
    raw.columns = [str(c).strip() for c in raw.columns]

    # Identify columns
    date_col = next((c for c in raw.columns if c.lower() in ("date", "datetime")), None)
    hour_col = next((c for c in raw.columns if c.lower() == "hour"), None)
    gen_col = next((c for c in raw.columns if "gen" in c.lower() or "mwh" in c.lower()), None)

    if date_col is None or gen_col is None:
        raise ValueError(f"Cannot identify Date and generation columns in {path.name}: {list(raw.columns)}")

    raw = raw[raw[date_col].apply(lambda v: not isinstance(v, str))].copy()
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col])

    if hour_col:
        raw[hour_col] = pd.to_numeric(raw[hour_col], errors="coerce")
        raw = raw.dropna(subset=[hour_col])
        # Hours are typically 1-24; convert to 0-23
        raw["_hour"] = raw[hour_col].astype(int).clip(1, 24) - 1
        timestamps = raw[date_col] + pd.to_timedelta(raw["_hour"], unit="h")
    else:
        timestamps = raw[date_col]

    raw[gen_col] = pd.to_numeric(raw[gen_col], errors="coerce").fillna(0.0)

    default_year = int(timestamps.dt.year.mode().iloc[0]) if not timestamps.empty else year

    # Derive a friendly project name from the filename
    stem = path.stem  # e.g. "VPPA_8760s_Liberty"
    # Strip the VPPA_8760s_ prefix if present
    for prefix in ("VPPA_8760s_", "VPPA_8760_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    profile_name = stem.replace("_", " ").strip() or path.stem

    hourly = pd.DataFrame({profile_name: raw[gen_col].values}, index=timestamps)
    hourly.index.name = "Time_Central"
    hourly = hourly.sort_index().groupby(level=0).first()

    return hourly, [profile_name], default_year, {}


@st.cache_data(show_spinner=False)
def load_vppa_summary_profiles_from_path(path_str: str, year: int, file_mtime_ns: int):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")
    # Try Summary sheet first; fall back to single-project format
    try:
        raw = pd.read_excel(path, sheet_name="Summary", header=None)
        return _parse_summary_sheet(raw, year)
    except Exception:
        return _parse_single_project_sheet(path, year)


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
    corr_granularity: str = "interval",
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
            if corr_granularity == "daily" and not pair.empty:
                pair = pair.groupby(pair.index.floor("D")).sum()
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
        if corr_granularity == "daily" and not pair.empty:
            pair = pair.groupby(pair.index.floor("D")).sum()
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
    # Purge ALL Streamlit caches whenever the code version changes.
    _cache_key = f"weather_cache_{_CODE_VERSION}"
    if _cache_key not in st.session_state:
        st.cache_data.clear()
        st.session_state[_cache_key] = True

    st.header("VPPA 8760 Comparison")

    # ── Mode picker ──────────────────────────────────────────────────────────
    mode = st.radio(
        "Mode",
        options=["Generate Weather Model Profiles", "Compare vs Bill Validation Model"],
        index=0,
        horizontal=True,
        key="vppa_8760_mode",
        help=(
            "Generate Weather Model Profiles: build an actual-weather or TMY profile for each XLS project "
            "from the registry (no curtailment, no SCED) and compare vs the XLS values.\n"
            "Compare vs Model: compare XLS profiles against a completed Bill Validation run."
        ),
    )

    # ── Workbook source ───────────────────────────────────────────────────────
    uploaded_wb = st.file_uploader(
        "VPPA Workbook (.xlsx/.xls/.xlsm)",
        type=["xlsx", "xls", "xlsm"],
        key="vppa_8760_workbook_upload",
        help="Upload a VPPA 8760 Excel workbook when the file is not available locally.",
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
    )

    # ── Load workbook ─────────────────────────────────────────────────────────
    try:
        source_map: dict = {}
        offtake_map: dict = {}
        if source_mode == "Upload":
            if uploaded_wb is None:
                st.error("No workbook uploaded yet. Upload an Excel file above.")
                return
            profile_df, profile_cols, wb_year, offtake_map = load_vppa_summary_profiles_from_bytes(
                uploaded_wb.getvalue(), 2025
            )
            source_name = uploaded_wb.name
        elif local_workbooks:
            workbook_specs = tuple(
                (str(p), int(p.stat().st_mtime_ns)) for p in local_workbooks
            )
            profile_df, profile_cols, wb_year, source_map, offtake_map = load_local_profile_catalog(
                workbook_specs, 2025
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

    # ── Profile selection ─────────────────────────────────────────────────────
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
        default=profile_cols,
        key="vppa_compare_selected_profiles",
        format_func=_format_profile_option,
    )
    include_total = st.checkbox(
        "Include total of selected profiles",
        value=len(selected_profiles) > 1,
        key="vppa_compare_include_total",
    )

    if not selected_profiles:
        st.warning("Select at least one VPPA profile.")
        return

    # ── Month filter ──────────────────────────────────────────────────────────
    _ALL_MONTHS = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    selected_month_names = st.multiselect(
        "Months",
        _ALL_MONTHS,
        default=_ALL_MONTHS,
        key="vppa_8760_month_select",
        help="Filter the comparison to specific months.",
    )
    _month_name_to_num = {m: i + 1 for i, m in enumerate(_ALL_MONTHS)}
    selected_month_nums = sorted(_month_name_to_num[m] for m in selected_month_names)
    if not selected_month_nums:
        st.warning("⚠️ No months selected. Please select at least one month.")
        return

    # Apply month filter to the XLS profile data
    if len(selected_month_nums) < 12:
        profile_df = profile_df[profile_df.index.month.isin(selected_month_nums)]

    # ══════════════════════════════════════════════════════════════════════════
    # MODE A — Generate weather-model profiles from registry
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "Generate Weather Model Profiles":
        st.markdown("---")

        # Weather source toggle
        weather_source = st.radio(
            "Weather data source",
            ["Actual Weather (Open-Meteo)", "TMY (PVGIS long-term average)"],
            index=0,
            horizontal=True,
            help="**Actual Weather** uses real historical weather for each year in the XLS. "
                 "**TMY** uses a typical meteorological year average.",
        )
        use_actual = weather_source.startswith("Actual")

        # Determine which years appear in the XLS data
        xls_years = sorted(profile_df.index.year.unique())
        if not xls_years:
            xls_years = [2025]

        source_label = "Actual Weather" if use_actual else "TMY"
        st.subheader(f"{source_label} Model Profiles")
        st.caption(
            f"Generating **{source_label.lower()}** profiles for each selected XLS project "
            f"(years auto-detected per project). "
            "**No curtailment** (negative-price curtailment is ignored). "
            "**No SCED** (pure weather model). "
            "Capacity is taken from the registry (`capacity_mw`)."
        )

        registry = _load_registry()
        if not registry:
            st.error(f"Registry not found at `{REGISTRY_PATH}`. Cannot generate profiles.")
            return

        # Build model series dict: {profile_col: hourly_MWh_series}
        model_series_map: dict[str, pd.Series] = {}
        meta_rows = []

        for proj in selected_profiles:
            meta = _match_project_meta(proj, registry)
            if meta is None:
                st.warning(f"⚠️ No registry match for **{proj}** — skipping.")
                continue

            p_name   = meta.get("project_name") or meta.get("name") or proj
            tech     = meta.get("tech", "Wind")
            lat      = float(meta.get("lat", 32.4))
            lon      = float(meta.get("lon", -99.7))
            cap_mw   = float(meta.get("capacity_mw", 100.0))
            t_model  = str(meta.get("turbine_model", "GENERIC"))
            hub_h    = float(meta.get("hub_height_m", 80.0))

            # Use offtake MW from XLS when available; fall back to full capacity
            offtake_mw = offtake_map.get(proj)
            use_offtake = offtake_mw is not None and pd.notna(offtake_mw) and float(offtake_mw) > 0
            scale = float(offtake_mw) / cap_mw if use_offtake else 1.0
            display_mw = float(offtake_mw) if use_offtake else cap_mw

            meta_rows.append({
                "XLS Profile": proj,
                "Registry Name": p_name,
                "Tech": tech,
                "Capacity (MW)": cap_mw,
                "Offtake (MW)": float(offtake_mw) if use_offtake else "—",
                "Lat": lat,
                "Lon": lon,
                "Turbine": t_model if tech == "Wind" else "—",
            })

            # Determine which years THIS project's XLS data actually covers
            proj_years = xls_years  # fallback to global
            if proj in profile_df.columns:
                proj_data = profile_df[proj].dropna()
                if not proj_data.empty:
                    proj_years = sorted(proj_data.index.year.unique())

            # Drop the current calendar year — Open-Meteo data is incomplete
            # for years still in progress, which inflates model totals.
            current_year = datetime.now().year
            proj_years = [y for y in proj_years if y < current_year]
            if not proj_years:
                st.warning(f"No complete calendar years in XLS for {p_name}; skipping model.")
                continue

            # Generate model for each year in the XLS and concatenate
            year_series = []
            for yr in proj_years:
                with st.spinner(f"Generating {source_label} {yr} for {p_name} ({tech}, {display_mw:.0f} MW)…"):
                    try:
                        s = _generate_weather_profile(
                            project_name=p_name,
                            tech=tech,
                            lat=lat,
                            lon=lon,
                            capacity_mw=cap_mw,
                            year=yr,
                            turbine_model=t_model,
                            hub_height_m=hub_h,
                            use_actual_weather=use_actual,
                        )
                        if not s.empty:
                            year_series.append(s)
                    except Exception as e:
                        st.warning(f"Error generating {source_label} {yr} for {p_name}: {e}")

            if year_series:
                combined_s = pd.concat(year_series).sort_index()
                combined_s = combined_s[~combined_s.index.duplicated(keep="first")]
                # Scale to offtake share if XLS provides an offtake MW
                if use_offtake and abs(scale - 1.0) > 1e-6:
                    combined_s = combined_s * scale
                model_series_map[proj] = combined_s
            else:
                st.error(f"Could not generate {source_label} for {p_name} — no data returned.")

        if not model_series_map:
            st.error(f"No {source_label} profiles could be generated.")
            return

        # Show registry metadata
        with st.expander("Registry metadata used", expanded=False):
            st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)

        # ── Compare each project individually ──────────────────────────────
        st.markdown("---")
        st.subheader("Per-Project Comparison")

        all_summary_rows = []

        for proj in selected_profiles:
            if proj not in model_series_map:
                continue

            model_s = model_series_map[proj]   # hourly MWh (avg MW over 1h)
            # Apply same month filter to model series
            if len(selected_month_nums) < 12:
                model_s = model_s[model_s.index.month.isin(selected_month_nums)]
            profile_s_raw = profile_df[proj]   # hourly MWh from XLS (already filtered)

            # Align on timestamps
            combined = pd.DataFrame({"Model_MWh": model_s, proj: profile_s_raw}).dropna()
            if combined.empty:
                st.warning(f"No overlapping timestamps for {proj}.")
                continue

            m_s = combined["Model_MWh"]
            p_s = combined[proj]

            row = _build_metrics(m_s, p_s)
            row["Profile"] = proj
            all_summary_rows.append(row)

        # ── Summary metrics table ──────────────────────────────────────────
        if all_summary_rows:
            mdf = pd.DataFrame(all_summary_rows)[[
                "Profile", "Hours Compared", "Model Total (MWh)",
                "Profile Total (MWh)", "Difference (MWh)", "Difference (%)",
                "MAE (MWh/h)", "RMSE (MWh/h)", "Correlation",
            ]]
            st.subheader("Summary Metrics")
            st.dataframe(
                mdf.style.format({
                    "Model Total (MWh)":   "{:,.0f}",
                    "Profile Total (MWh)": "{:,.0f}",
                    "Difference (MWh)":    "{:+,.0f}",
                    "Difference (%)":      "{:+.2f}%",
                    "MAE (MWh/h)":         "{:,.2f}",
                    "RMSE (MWh/h)":        "{:,.2f}",
                    "Correlation":         "{:.3f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # ── Monthly energy comparison table & chart ────────────────────────
        # Build a combined compare_table with one Model_MWh column (sum of all models)
        # and individual XLS profile columns, indexed by timestamp.
        all_model_cols = {}
        all_profile_cols_data = {}
        for proj in selected_profiles:
            if proj not in model_series_map:
                continue
            all_model_cols[proj] = model_series_map[proj]
            all_profile_cols_data[proj] = profile_df[proj]

        if all_model_cols:
            # Build combined frame for monthly bar chart
            combined_all = pd.DataFrame(all_model_cols).join(
                pd.DataFrame(all_profile_cols_data), how="outer", lsuffix="_Model", rsuffix="_XLS"
            )
            combined_all.index = pd.to_datetime(combined_all.index)
            combined_all["Month"] = combined_all.index.to_period("M").astype(str)

            # Build a per-project monthly summary
            monthly_rows_all = []
            for proj in selected_profiles:
                model_col = proj + "_Model" if (proj + "_Model") in combined_all.columns else proj
                xls_col   = proj + "_XLS"   if (proj + "_XLS")   in combined_all.columns else proj
                if model_col not in combined_all.columns or xls_col not in combined_all.columns:
                    continue
                ml = combined_all.groupby("Month")[model_col].sum()
                xl = combined_all.groupby("Month")[xls_col].sum()
                for mo in sorted(set(ml.index) | set(xl.index)):
                    monthly_rows_all.append({
                        "Project": proj,
                        "Month": mo,
                        f"Model {source_label} (MWh)": ml.get(mo, np.nan),
                        "XLS (MWh)": xl.get(mo, np.nan),
                    })

            if monthly_rows_all:
                monthly_df_all = pd.DataFrame(monthly_rows_all)
                model_col_name = f"Model {source_label} (MWh)"
                monthly_df_all["Diff (MWh)"] = monthly_df_all[model_col_name] - monthly_df_all["XLS (MWh)"]
                monthly_df_all["Diff (%)"] = np.where(
                    monthly_df_all["XLS (MWh)"] != 0,
                    (monthly_df_all["Diff (MWh)"] / monthly_df_all["XLS (MWh)"]) * 100.0,
                    np.nan,
                )

                # Append a TOTAL row per project
                total_rows = []
                for proj in monthly_df_all["Project"].unique():
                    sub = monthly_df_all[monthly_df_all["Project"] == proj]
                    m_total = sub[model_col_name].sum()
                    x_total = sub["XLS (MWh)"].sum()
                    diff_mwh = m_total - x_total
                    diff_pct = (diff_mwh / x_total * 100.0) if x_total != 0 else np.nan
                    total_rows.append({
                        "Project": proj,
                        "Month": "TOTAL",
                        model_col_name: m_total,
                        "XLS (MWh)": x_total,
                        "Diff (MWh)": diff_mwh,
                        "Diff (%)": diff_pct,
                    })
                monthly_df_all = pd.concat(
                    [monthly_df_all, pd.DataFrame(total_rows)], ignore_index=True,
                )

                st.subheader("Monthly Energy Breakdown")
                st.dataframe(
                    monthly_df_all.style.format({
                        model_col_name: "{:,.0f}",
                        "XLS (MWh)":       "{:,.0f}",
                        "Diff (MWh)":      "{:+,.0f}",
                        "Diff (%)":        "{:+.2f}%",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                # Bar chart
                fig = go.Figure()
                colors = ["#3A86FF", "#FF6B6B", "#FFBE0B", "#8338EC", "#06D6A0"]
                for i, proj in enumerate(selected_profiles):
                    sub = monthly_df_all[monthly_df_all["Project"] == proj]
                    if sub.empty:
                        continue
                    c = colors[i % len(colors)]
                    fig.add_trace(go.Bar(
                        name=f"{proj} — Model {source_label}",
                        x=sub["Month"],
                        y=sub[model_col_name],
                        marker_color=c,
                    ))
                    fig.add_trace(go.Bar(
                        name=f"{proj} — XLS",
                        x=sub["Month"],
                        y=sub["XLS (MWh)"],
                        marker_color=c,
                        marker_pattern_shape="/",
                    ))
                fig.update_layout(
                    barmode="group",
                    title=f"Monthly Energy: Model {source_label} vs XLS (MWh)",
                    xaxis_title="Month",
                    yaxis_title="MWh",
                    legend_title="Series",
                )
                st.plotly_chart(fig, use_container_width=True)

        return  # end Mode A (weather model)

    # ══════════════════════════════════════════════════════════════════════════
    # MODE B — Compare vs Bill Validation model (original flow)
    # ══════════════════════════════════════════════════════════════════════════
    if "val_preview_results" not in st.session_state:
        st.info(
            "Run a model in **Bill Validation** first, then come back here to compare "
            "against the VPPA 8760 profiles. Or switch Mode to **Generate Weather Model Profiles** above."
        )
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
        )
    with c4:
        time_alignment_label = st.selectbox(
            "Time Alignment",
            options=["CST (No DST)", "US/Central (DST-aware)"],
            index=0,
            key="vppa_compare_time_alignment",
        )
    with c5:
        energy_basis_label = st.selectbox(
            "Model Energy Basis",
            options=["Settled (post-curtail)", "Potential (pre-curtail)"],
            index=1,
            key="vppa_compare_energy_basis",
        )
    time_alignment = "CST_NO_DST" if time_alignment_label == "CST (No DST)" else "US_CENTRAL_DST"
    energy_basis = "POTENTIAL" if energy_basis_label == "Potential (pre-curtail)" else "SETTLED"

    if compare_interval == "15-min":
        st.caption("15-min mode distributes each hourly VPPA value evenly across four quarter-hour intervals.")
    if time_alignment == "CST_NO_DST":
        st.caption("Using fixed CST (no DST) alignment for model timestamps.")
    if energy_basis == "POTENTIAL":
        st.caption("Comparing against potential generation (curtailment added back).")

    profile_shift_hours = st.number_input(
        "VPPA Time Shift (hours)", min_value=-24, max_value=24, value=0, step=1,
        key="vppa_compare_profile_shift_hours",
    )

    model_df = st.session_state["val_preview_results"][model_name]
    model_series = _extract_model_series(
        model_df, compare_interval, time_alignment=time_alignment, energy_basis=energy_basis,
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

    count_label  = "Hours Compared" if compare_interval == "Hourly" else "Intervals Compared"
    error_label  = "MAE (MWh/h)"    if compare_interval == "Hourly" else "MAE (MWh/interval)"
    rmse_label   = "RMSE (MWh/h)"   if compare_interval == "Hourly" else "RMSE (MWh/interval)"
    metric_rows = []
    compare_targets = selected_profiles + (["Selected Total"] if include_total else [])
    for target in compare_targets:
        joined = compare_table[["Model_MWh", target]].dropna()
        if joined.empty:
            continue
        metrics = _build_metrics(joined["Model_MWh"], joined[target])
        metrics[count_label] = metrics.pop("Hours Compared")
        metrics[error_label] = metrics.pop("MAE (MWh/h)")
        metrics[rmse_label]  = metrics.pop("RMSE (MWh/h)")
        metrics["Profile"] = target
        metric_rows.append(metrics)

    if not metric_rows:
        st.error("No overlapping timestamps between model data and selected VPPA profiles.")
        return

    mdf = pd.DataFrame(metric_rows)[[
        "Profile", count_label, "Model Total (MWh)", "Profile Total (MWh)",
        "Difference (MWh)", "Difference (%)", error_label, rmse_label, "Correlation",
    ]]
    st.subheader("Summary Metrics")
    st.dataframe(
        mdf.style.format({
            "Model Total (MWh)":   "{:,.0f}",
            "Profile Total (MWh)": "{:,.0f}",
            "Difference (MWh)":    "{:+,.0f}",
            "Difference (%)":      "{:+.2f}%",
            error_label:           "{:,.2f}",
            rmse_label:            "{:,.2f}",
            "Correlation":         "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    lag_rows = []
    for target in selected_profiles:
        joined = compare_table[["Model_MWh", target]].dropna()
        if joined.empty:
            continue
        best = _best_lag_correlation(joined["Model_MWh"], joined[target], interval_label=compare_interval, max_hours=24)
        lag_rows.append({"Profile": target, "Best Lag (hours)": best["Lag (hours)"], "Best Correlation": best["Correlation"], "Points": best["Points"]})

    if lag_rows:
        with st.expander("Lag Diagnostic (best correlation within +/-24 hours)", expanded=False):
            st.dataframe(
                pd.DataFrame(lag_rows).style.format({"Best Lag (hours)": "{:+.2f}", "Best Correlation": "{:.3f}", "Points": "{:,.0f}"}),
                use_container_width=True, hide_index=True,
            )

    monthly_corr_targets = selected_profiles + (["Selected Total"] if include_total else [])
    monthly_corr_count_label = "Hours Compared" if compare_interval == "Hourly" else "Intervals Compared"
    monthly_corr_df = _build_monthly_correlation(compare_table, monthly_corr_targets, count_label=monthly_corr_count_label, corr_granularity="interval")
    if not monthly_corr_df.empty:
        corr_fmt = {c: "{:.3f}" for c in monthly_corr_targets if c in monthly_corr_df.columns}
        st.subheader("Monthly Correlation (Pearson R)")
        interval_caption = "hourly" if compare_interval == "Hourly" else "15-min"
        st.caption(f"Correlation is computed from {interval_caption} MWh intervals within each month.")
        st.dataframe(monthly_corr_df.style.format(corr_fmt), use_container_width=True, hide_index=True)

    monthly = compare_table.copy()
    monthly["Month"] = monthly.index.to_period("M").astype(str)
    chart_targets = selected_profiles
    agg_cols = ["Model_MWh"] + chart_targets
    monthly_agg = monthly.groupby("Month", as_index=False)[agg_cols].sum()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Model", x=monthly_agg["Month"], y=monthly_agg["Model_MWh"]))
    for target in chart_targets:
        fig.add_trace(go.Bar(name=target, x=monthly_agg["Month"], y=monthly_agg[target]))
    fig.update_layout(barmode="group", title="Monthly Energy Comparison (MWh)", xaxis_title="Month", yaxis_title="MWh", legend_title="Series")
    st.plotly_chart(fig, use_container_width=True)

    monthly_table_cols = ["Model_MWh"] + selected_profiles + (["Selected Total"] if include_total else [])
    monthly_table = monthly.groupby("Month", as_index=False)[monthly_table_cols].sum()
    monthly_table = monthly_table.rename(columns={"Model_MWh": "Model"})
    table_targets = selected_profiles + (["Selected Total"] if include_total else [])
    corr_lookup = pd.DataFrame()
    if not monthly_corr_df.empty:
        corr_lookup = monthly_corr_df[monthly_corr_df["Month"] != "Total"].set_index("Month")

    for target in table_targets:
        if target not in monthly_table.columns:
            continue
        diff_col = f"{target} Diff (MWh)"
        pct_col  = f"{target} Diff (%)"
        corr_col = f"{target} Monthly Corr (R)"
        monthly_table[diff_col] = monthly_table[target] - monthly_table["Model"]
        monthly_table[pct_col]  = np.where(monthly_table["Model"] != 0, (monthly_table[diff_col] / monthly_table["Model"]) * 100.0, np.nan)
        if not corr_lookup.empty and target in corr_lookup.columns:
            monthly_table[corr_col] = monthly_table["Month"].map(corr_lookup[target])
        else:
            monthly_table[corr_col] = np.nan

    total_row = {"Month": "Total", "Model": float(compare_table["Model_MWh"].sum())}
    for target in table_targets:
        if target not in compare_table.columns:
            continue
        pair = compare_table[["Model_MWh", target]].dropna()
        target_total = float(pair[target].sum()) if not pair.empty else np.nan
        total_diff   = target_total - total_row["Model"] if pd.notna(target_total) else np.nan
        total_pct    = (total_diff / total_row["Model"]) * 100.0 if total_row["Model"] != 0 else np.nan
        total_row[target] = target_total
        total_row[f"{target} Diff (MWh)"] = total_diff
        total_row[f"{target} Diff (%)"]   = total_pct
        total_row[f"{target} Monthly Corr (R)"] = np.nan

    monthly_table = pd.concat([monthly_table, pd.DataFrame([total_row])], ignore_index=True)

    ordered_cols = ["Month", "Model"]
    for target in table_targets:
        ordered_cols.extend([target, f"{target} Diff (MWh)", f"{target} Diff (%)", f"{target} Monthly Corr (R)"])
    monthly_table = monthly_table[[c for c in ordered_cols if c in monthly_table.columns]]

    fmt_map = {}
    for c in monthly_table.columns:
        if c == "Month":
            continue
        if c.endswith("Diff (MWh)"):
            fmt_map[c] = "{:+,.0f}"
        elif c.endswith("Diff (%)"):
            fmt_map[c] = "{:+.2f}%"
        elif c.endswith("Corr (R)"):
            fmt_map[c] = "{:.3f}"
        else:
            fmt_map[c] = "{:,.0f}"

    st.subheader("Monthly Energy Table (MWh)")
    interval_caption = "hourly" if compare_interval == "Hourly" else "15-min"
    st.caption(f"`Monthly Corr (R)` columns use {interval_caption} intervals within each month.")
    st.dataframe(monthly_table.style.format(fmt_map), use_container_width=True, hide_index=True)

    with st.expander("View hourly comparison data"):
        st.dataframe(compare_table, use_container_width=True)

