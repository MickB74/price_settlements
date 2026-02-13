import json
import re
import statistics
import zipfile
from calendar import monthrange
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

# Canonical hub names used inside calibration helpers.
CANONICAL_HUBS = ("NORTH", "SOUTH", "WEST", "HOUSTON", "PAN")

HUB_COORDS = {
    "NORTH": (32.3865, -96.8475),
    "SOUTH": (26.9070, -99.2715),
    "WEST": (32.4518, -100.5371),
    "HOUSTON": (29.3013, -94.7977),
    "PAN": (35.2220, -101.8313),
}

HUB_ALIASES = {
    "HB_NORTH": "NORTH",
    "HB_SOUTH": "SOUTH",
    "HB_WEST": "WEST",
    "HB_HOUSTON": "HOUSTON",
    "HB_PAN": "PAN",
    "NORTH": "NORTH",
    "SOUTH": "SOUTH",
    "WEST": "WEST",
    "HOUSTON": "HOUSTON",
    "PAN": "PAN",
}

# Hub-level defaults tuned toward reducing observed wind underprediction.
DEFAULT_HUB_SHEAR_ALPHA = {
    "NORTH": 0.34,
    "SOUTH": 0.33,
    "WEST": 0.31,
    "HOUSTON": 0.24,
    "PAN": 0.32,
}

DEFAULT_HUB_MULTIPLIER = {
    "NORTH": 1.10,
    "SOUTH": 1.07,
    "WEST": 1.02,
    "HOUSTON": 1.03,
    "PAN": 1.04,
}

# Annual zonal capacity-factor anchors used to build month-level CF targets.
DEFAULT_ANNUAL_CF_BY_HUB = {
    "NORTH": 0.40,
    "SOUTH": 0.37,
    "WEST": 0.43,
    "HOUSTON": 0.36,
    "PAN": 0.45,
}

# Texas wind shape: stronger spring/winter, weaker summer shoulder.
DEFAULT_MONTHLY_CF_SHAPE = {
    1: 1.08,
    2: 1.03,
    3: 1.13,
    4: 1.18,
    5: 1.08,
    6: 0.90,
    7: 0.82,
    8: 0.80,
    9: 0.86,
    10: 0.95,
    11: 1.04,
    12: 1.13,
}

DEFAULT_REANALYSIS_BLEND = {
    "era5_weight": 0.85,
    "merra2_weight": 0.15,
}

DEFAULT_POSTPROCESS_CONFIG = {
    "apply_monthly_targets": False,
    "apply_sced_bias": True,
    "apply_node_adjustment": True,
    "apply_congestion_haircut": False,
}

# Congestion-zone haircut factors used only during low-price intervals.
DEFAULT_CONGESTION_HAIRCUT = {
    "price_trigger": 15.0,
    "deep_price_trigger": 0.0,
    "by_hub": {
        "NORTH": 0.94,
        "SOUTH": 0.97,
        "WEST": 0.90,
        "HOUSTON": 0.98,
        "PAN": 0.89,
    },
    "deep_by_hub": {
        "NORTH": 0.78,
        "SOUTH": 0.86,
        "WEST": 0.72,
        "HOUSTON": 0.88,
        "PAN": 0.70,
    },
    "resource_multiplier": {},
}

EIA_SHEET_PATH = "xl/worksheets/sheet1.xml"
EIA_SHARED_STRINGS_PATH = "xl/sharedStrings.xml"
XML_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

NAME_STOPWORDS = {
    "wind",
    "energy",
    "farm",
    "project",
    "plant",
    "power",
    "llc",
    "co",
    "lp",
    "the",
    "unit",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _as_float(value):
    try:
        out = float(value)
        return out
    except (TypeError, ValueError):
        return None


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    txt = str(value).strip().lower()
    if txt in {"1", "true", "yes", "y", "on"}:
        return True
    if txt in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _clamp(val, low, high):
    return float(max(low, min(high, val)))


def _month_key_candidates(month_num):
    month_num = int(month_num)
    month_name = datetime(2000, month_num, 1).strftime("%B").lower()
    month_abbr = datetime(2000, month_num, 1).strftime("%b").lower()
    return (
        str(month_num),
        f"{month_num:02d}",
        month_name,
        month_abbr,
    )


def _lookup_month_value(mapping, month_num):
    if not isinstance(mapping, dict):
        return None
    month_keys = {str(k).strip().lower(): v for k, v in mapping.items()}
    for key in _month_key_candidates(month_num):
        if key in month_keys:
            return _as_float(month_keys[key])
    return None


def _safe_series(values, index=None):
    if isinstance(values, pd.Series):
        return values.astype(float)
    if index is None:
        return pd.Series(values, dtype=float)
    return pd.Series(values, index=index, dtype=float)


def _infer_interval_hours(index):
    if len(index) < 2:
        return 0.25
    try:
        diffs = np.diff(index.asi8) / 1e9
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return 0.25
        return float(np.median(diffs) / 3600.0)
    except Exception:
        return 0.25


def _default_monthly_cf_targets():
    out = {}
    for hub, annual_cf in DEFAULT_ANNUAL_CF_BY_HUB.items():
        hub_targets = {}
        for month, shape in DEFAULT_MONTHLY_CF_SHAPE.items():
            hub_targets[str(month)] = round(_clamp(annual_cf * shape, 0.05, 0.70), 4)
        out[hub] = hub_targets
    return out


def _deepcopy_jsonable(obj):
    return json.loads(json.dumps(obj))


def _cell_col(ref):
    m = re.match(r"([A-Z]+)\d+$", str(ref))
    return m.group(1) if m else None


def _coerce_num(val):
    if val is None:
        return None
    txt = str(val).strip()
    if txt in {"", ".", "..", "NA", "N/A"}:
        return None
    txt = txt.replace(",", "")
    try:
        return float(txt)
    except ValueError:
        return None


def _read_shared_strings(zf):
    if EIA_SHARED_STRINGS_PATH not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(EIA_SHARED_STRINGS_PATH))
    out = []
    for si in root.findall("a:si", XML_NS):
        text = "".join(t.text or "" for t in si.findall(".//a:t", XML_NS))
        out.append(text)
    return out


def _parse_sheet_rows(zf, sheet_path):
    shared = _read_shared_strings(zf)
    ws = ET.fromstring(zf.read(sheet_path))
    sheet_data = ws.find("a:sheetData", XML_NS)
    if sheet_data is None:
        return []

    rows = []
    for row in sheet_data.findall("a:row", XML_NS):
        row_num = int(row.attrib.get("r", "0"))
        vals = {}
        for cell in row.findall("a:c", XML_NS):
            ref = cell.attrib.get("r")
            col = _cell_col(ref)
            if not col:
                continue
            t = cell.attrib.get("t")
            v = cell.find("a:v", XML_NS)
            if v is None:
                vals[col] = ""
                continue
            raw = v.text
            if t == "s" and raw is not None:
                try:
                    raw = shared[int(raw)]
                except Exception:
                    raw = ""
            vals[col] = raw if raw is not None else ""
        rows.append((row_num, vals))
    return rows


def _parse_eia923_tx_wind_monthly(eia_path):
    """
    Returns map keyed by plant id:
    {
      1234: {"plant_name": "...", "months": {1: mwh, ... 12: mwh}}
    }
    """
    if not eia_path or not Path(eia_path).exists():
        return {}

    out = {}
    month_cols = {
        1: "CB",
        2: "CC",
        3: "CD",
        4: "CE",
        5: "CF",
        6: "CG",
        7: "CH",
        8: "CI",
        9: "CJ",
        10: "CK",
        11: "CL",
        12: "CM",
    }
    with zipfile.ZipFile(eia_path) as zf:
        rows = _parse_sheet_rows(zf, EIA_SHEET_PATH)

    plant_rows = defaultdict(lambda: {"plant_name": None, "months": defaultdict(float)})
    for row_num, vals in rows:
        if row_num < 7:
            continue
        state = str(vals.get("G", "")).strip().upper()
        if state != "TX":
            continue

        fuel = str(vals.get("O", "")).strip().upper()
        prime = str(vals.get("N", "")).strip().upper()
        if fuel != "WND" and prime not in {"WT", "WS"}:
            continue

        plant_id_val = _coerce_num(vals.get("A"))
        if plant_id_val is None:
            continue
        plant_id = int(plant_id_val)

        entry = plant_rows[plant_id]
        plant_name = str(vals.get("D", "")).strip()
        if plant_name and not entry["plant_name"]:
            entry["plant_name"] = plant_name

        for month, col in month_cols.items():
            mwh = _coerce_num(vals.get(col))
            if mwh is not None:
                entry["months"][month] += float(max(0.0, mwh))

    for plant_id, data in plant_rows.items():
        out[plant_id] = {
            "plant_name": data["plant_name"] or f"Plant {plant_id}",
            "months": {m: float(data["months"].get(m, 0.0)) for m in range(1, 13)},
        }
    return out


def _normalize_name_for_match(name):
    base = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    toks = [t for t in base.split() if t and t not in NAME_STOPWORDS]
    return " ".join(toks)


def _name_similarity(left, right):
    l_raw = str(left or "").lower().strip()
    r_raw = str(right or "").lower().strip()
    if not l_raw or not r_raw:
        return 0.0

    l_norm = _normalize_name_for_match(l_raw)
    r_norm = _normalize_name_for_match(r_raw)
    if not l_norm or not r_norm:
        return 0.0

    ratio = SequenceMatcher(None, l_norm, r_norm).ratio()
    l_set = set(l_norm.split())
    r_set = set(r_norm.split())
    inter = len(l_set & r_set)
    union = len(l_set | r_set) if (l_set or r_set) else 1
    jaccard = inter / union

    if l_norm in r_norm or r_norm in l_norm:
        ratio = max(ratio, 0.92)
    return float(0.65 * ratio + 0.35 * jaccard)


def _collect_unique_wind_assets(assets):
    unique = {}
    for project_name, meta in assets.items():
        if str(meta.get("tech", "")).lower() != "wind":
            continue
        resource = str(meta.get("resource_name", "")).strip().upper()
        if not resource:
            continue
        existing = unique.get(resource)
        candidate = {
            "resource_id": resource,
            "project_names": [project_name],
            "hub": normalize_hub_name(meta.get("hub")),
            "capacity_mw": _as_float(meta.get("capacity_mw")) or 0.0,
        }
        proj_alias = str(meta.get("project_name", "")).strip()
        if proj_alias and proj_alias not in candidate["project_names"]:
            candidate["project_names"].append(proj_alias)

        if existing is None:
            unique[resource] = candidate
            continue

        for nm in candidate["project_names"]:
            if nm not in existing["project_names"]:
                existing["project_names"].append(nm)
        if (existing.get("hub") is None) and candidate.get("hub"):
            existing["hub"] = candidate["hub"]
        existing["capacity_mw"] = max(existing.get("capacity_mw", 0.0), candidate["capacity_mw"])
    return unique


def _match_assets_to_eia_plants(unique_assets, eia_plants, score_threshold=0.62):
    candidates = []
    for rid, meta in unique_assets.items():
        for pid, pdata in eia_plants.items():
            best_score = 0.0
            for nm in meta.get("project_names", []):
                score = _name_similarity(nm, pdata.get("plant_name"))
                if score > best_score:
                    best_score = score
            if best_score >= score_threshold:
                candidates.append((best_score, rid, pid))

    candidates.sort(reverse=True)
    assigned_resources = set()
    assigned_plants = set()
    matches = {}
    for score, rid, pid in candidates:
        if rid in assigned_resources or pid in assigned_plants:
            continue
        matches[rid] = {"plant_id": pid, "score": round(score, 4)}
        assigned_resources.add(rid)
        assigned_plants.add(pid)
    return matches


def _hours_in_month(year):
    return {m: monthrange(year, m)[1] * 24.0 for m in range(1, 13)}


def _load_cached_asset_period_data(resource_id, start_date, end_date, cache_dir):
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    years = range(start.year, end.year + 1)
    frames = []

    for year in years:
        file_path = Path(cache_dir) / f"{resource_id}_{year}_full.parquet"
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
            except Exception:
                df = pd.DataFrame()
            if not df.empty and "Time" in df.columns:
                df = df.copy()
                df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
                df = df.dropna(subset=["Time"])
                mask = (df["Time"].dt.date >= start) & (df["Time"].dt.date <= end)
                frames.append(df.loc[mask])
                continue

        y_start = max(start, datetime(year, 1, 1).date())
        y_end = min(end, datetime(year, 12, 31).date())
        for day in pd.date_range(y_start, y_end, freq="D"):
            daily_file = Path(cache_dir) / f"{day.date()}_{resource_id}.parquet"
            if not daily_file.exists():
                continue
            try:
                ddf = pd.read_parquet(daily_file)
            except Exception:
                continue
            if ddf.empty or "Time" not in ddf.columns:
                continue
            ddf = ddf.copy()
            ddf["Time"] = pd.to_datetime(ddf["Time"], utc=True, errors="coerce")
            ddf = ddf.dropna(subset=["Time"])
            frames.append(ddf)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("Time").sort_values("Time")


def _iter_unique_wind_asset_meta(assets):
    by_resource = {}
    for project_name, meta in assets.items():
        if str(meta.get("tech", "")).lower() != "wind":
            continue
        resource_id = str(meta.get("resource_name", "")).strip().upper()
        if not resource_id:
            continue

        cap = _as_float(meta.get("capacity_mw")) or 0.0
        candidate = {
            "project_name": project_name,
            "resource_id": resource_id,
            "hub": meta.get("hub"),
            "lat": _as_float(meta.get("lat")),
            "lon": _as_float(meta.get("lon")),
            "capacity_mw": cap,
            "hub_height_m": _as_float(meta.get("hub_height_m")) or 80.0,
            "turbine_manuf": meta.get("turbine_manuf", "GENERIC"),
            "turbine_model": meta.get("turbine_model", "GENERIC"),
            "rotor_diameter_m": _as_float(meta.get("rotor_diameter_m")),
            "turbines": meta.get("turbines"),
        }

        existing = by_resource.get(resource_id)
        if existing is None:
            by_resource[resource_id] = candidate
            continue

        existing_has_turbines = bool(existing.get("turbines"))
        candidate_has_turbines = bool(candidate.get("turbines"))
        take_candidate = False
        if candidate_has_turbines and not existing_has_turbines:
            take_candidate = True
        elif (candidate["capacity_mw"] > existing.get("capacity_mw", 0.0)) and (
            candidate_has_turbines == existing_has_turbines
        ):
            take_candidate = True

        if take_candidate:
            by_resource[resource_id] = candidate
        else:
            if (not existing.get("hub")) and candidate.get("hub"):
                existing["hub"] = candidate["hub"]
            if existing.get("lat") is None and candidate.get("lat") is not None:
                existing["lat"] = candidate["lat"]
            if existing.get("lon") is None and candidate.get("lon") is not None:
                existing["lon"] = candidate["lon"]

    return by_resource


def _derive_sced_month_hour_bias(table, assets, repo_root, start_date="2024-12-01", end_date="2025-11-30"):
    """
    Learn month-hour residual multipliers by hub using cached SCED + raw physics model.
    Cache-only by design to avoid network dependency.
    """
    try:
        import fetch_tmy
        from utils import power_curves
    except Exception:
        return table

    cache_dir = Path(repo_root) / "sced_cache"
    if not cache_dir.exists():
        return table

    by_resource = _iter_unique_wind_asset_meta(assets)
    if not by_resource:
        return table

    mh_values = defaultdict(lambda: defaultdict(list))  # hub -> (month, hour) -> [ratios]
    h_values = defaultdict(lambda: defaultdict(list))   # hub -> hour -> [ratios]
    resources_used = []
    total_points = 0

    for resource_id, meta in by_resource.items():
        lat = meta.get("lat")
        lon = meta.get("lon")
        cap = _as_float(meta.get("capacity_mw")) or 0.0
        if cap <= 0 or lat is None or lon is None:
            continue

        actual_df = _load_cached_asset_period_data(resource_id, start_date, end_date, cache_dir)
        if actual_df.empty or "Actual_MW" not in actual_df.columns:
            continue

        hub_height = _as_float(meta.get("hub_height_m")) or 80.0
        tech_type = power_curves.get_curve_for_specs(
            meta.get("turbine_manuf", "GENERIC"),
            meta.get("turbine_model", "GENERIC"),
            meta.get("rotor_diameter_m"),
        )

        turbines = meta.get("turbines")
        p24 = fetch_tmy.get_profile_for_year(
            2024,
            "Wind",
            cap,
            lat=lat,
            lon=lon,
            hub_height=hub_height,
            turbine_type=tech_type,
            hub_name=meta.get("hub"),
            project_name=meta.get("project_name"),
            resource_id=resource_id,
            apply_wind_calibration=False,
            turbines=turbines,
        )
        p25 = fetch_tmy.get_profile_for_year(
            2025,
            "Wind",
            cap,
            lat=lat,
            lon=lon,
            hub_height=hub_height,
            turbine_type=tech_type,
            hub_name=meta.get("hub"),
            project_name=meta.get("project_name"),
            resource_id=resource_id,
            apply_wind_calibration=False,
            turbines=turbines,
        )
        modeled = pd.concat([p24, p25])
        modeled = modeled[~modeled.index.duplicated(keep="first")]
        if modeled.empty:
            continue

        actual = actual_df.set_index("Time")["Actual_MW"]
        actual = pd.to_numeric(actual, errors="coerce")
        merged = pd.DataFrame(
            {
                "actual": actual,
                "modeled": modeled.reindex(actual.index),
            }
        ).dropna()
        if merged.empty:
            continue

        if "Base_Point_MW" in actual_df.columns:
            bp = pd.to_numeric(actual_df.set_index("Time")["Base_Point_MW"], errors="coerce")
            bp = bp.reindex(merged.index).fillna(np.inf).clip(lower=0.0)
            merged["modeled"] = np.minimum(merged["modeled"].values, bp.values)

        threshold = get_offline_threshold_mw(cap)
        valid = ~((merged["actual"] < 0.5) & (merged["modeled"] > threshold))
        valid &= merged["modeled"] > max(2.0, cap * 0.03)
        merged = merged.loc[valid]
        if len(merged) < 500:
            continue

        merged["ratio"] = (merged["actual"] / merged["modeled"]).clip(0.5, 1.7)
        merged["month"] = merged.index.month
        merged["hour"] = merged.index.hour
        hub = normalize_hub_name(meta.get("hub"))
        if not hub:
            continue

        for (month, hour), chunk in merged.groupby(["month", "hour"]):
            vals = chunk["ratio"].values.tolist()
            if vals:
                mh_values[hub][(int(month), int(hour))].extend(vals)
                h_values[hub][int(hour)].extend(vals)
                total_points += len(vals)

        resources_used.append(resource_id)

    if not resources_used:
        return table

    min_points_mh = 40
    min_points_h = 200
    mh_out = {}
    h_out = {}
    for hub in CANONICAL_HUBS:
        hub_h = {}
        for hour in range(24):
            vals = h_values.get(hub, {}).get(hour, [])
            if len(vals) >= min_points_h:
                hub_h[str(hour)] = round(_clamp(float(np.median(vals)), 0.75, 1.25), 4)
            else:
                hub_h[str(hour)] = 1.0
        h_out[hub] = hub_h

        hub_mh = {}
        for month in range(1, 13):
            for hour in range(24):
                vals = mh_values.get(hub, {}).get((month, hour), [])
                if len(vals) >= min_points_mh:
                    mult = _clamp(float(np.median(vals)), 0.75, 1.25)
                else:
                    mult = _as_float(hub_h.get(str(hour))) or 1.0
                hub_mh[f"{month}-{hour}"] = round(mult, 4)
        mh_out[hub] = hub_mh

    table["sced_bias_hourly_multiplier"] = h_out
    table["sced_bias_month_hour_multiplier"] = mh_out
    table["sced_bias_training"] = {
        "start_date": start_date,
        "end_date": end_date,
        "resources_used": sorted(set(resources_used)),
        "resource_count": len(set(resources_used)),
        "points_used": int(total_points),
        "cache_only": True,
    }
    table["source"] = f"{table.get('source', 'defaults')}+sced_shape"
    return table


def _find_latest_eia923_generation_workbook(root):
    files = sorted(root.glob("EIA923_Schedules_2_3_4_5_M_12_*_Final.xlsx"))
    if not files:
        return None, None
    parsed = []
    for p in files:
        m = re.search(r"_M_12_(\d{4})_Final\.xlsx$", p.name)
        year = int(m.group(1)) if m else 0
        parsed.append((year, p))
    parsed.sort(key=lambda x: x[0])
    year, path = parsed[-1]
    return (path if year > 0 else files[-1]), (year if year > 0 else None)


def _apply_eia923_monthly_targets(table, assets, eia_path, eia_year):
    eia_plants = _parse_eia923_tx_wind_monthly(eia_path)
    if not eia_plants:
        return table

    unique_assets = _collect_unique_wind_assets(assets)
    matches = _match_assets_to_eia_plants(unique_assets, eia_plants)
    if not matches:
        return table

    monthly_energy_target = {
        "projects": {},
        "resources": {},
        "hubs": {},
    }

    default_cf = _default_monthly_cf_targets()
    monthly_cf_target = _deepcopy_jsonable(default_cf)

    hub_mwh = defaultdict(lambda: defaultdict(float))
    hub_cap_matched = defaultdict(float)
    hub_cap_total = defaultdict(float)

    for rid, meta in unique_assets.items():
        hub = meta.get("hub")
        cap = _as_float(meta.get("capacity_mw")) or 0.0
        if hub and cap > 0:
            hub_cap_total[hub] += cap

    for rid, m in matches.items():
        meta = unique_assets.get(rid, {})
        plant = eia_plants.get(m["plant_id"])
        if not plant:
            continue
        months = {str(k): round(float(v), 2) for k, v in plant["months"].items()}
        monthly_energy_target["resources"][rid] = months

        for proj_name in meta.get("project_names", []):
            monthly_energy_target["projects"][proj_name] = months

        hub = meta.get("hub")
        cap = _as_float(meta.get("capacity_mw")) or 0.0
        if hub:
            for k, v in plant["months"].items():
                hub_mwh[hub][k] += float(v)
            if cap > 0:
                hub_cap_matched[hub] += cap

    for hub, by_month in hub_mwh.items():
        monthly_energy_target["hubs"][hub] = {str(m): round(by_month.get(m, 0.0), 2) for m in range(1, 13)}

    year = int(eia_year) if eia_year else 2024
    hours_map = _hours_in_month(year)
    for hub, by_month in hub_mwh.items():
        cap = hub_cap_matched.get(hub, 0.0)
        if cap <= 0:
            continue
        coverage = (hub_cap_matched.get(hub, 0.0) / hub_cap_total.get(hub, cap)) if hub_cap_total.get(hub, 0.0) > 0 else 1.0
        if coverage < 0.35:
            continue
        monthly_cf_target[hub] = {}
        for month in range(1, 13):
            cf = by_month.get(month, 0.0) / (cap * hours_map[month])
            monthly_cf_target[hub][str(month)] = round(_clamp(cf, 0.05, 0.70), 4)

    table["monthly_energy_target_mwh"] = monthly_energy_target
    table["monthly_cf_target"] = monthly_cf_target
    table["eia923_calibration"] = {
        "source_file": str(Path(eia_path).name),
        "year": year,
        "tx_wind_plants_found": len(eia_plants),
        "matched_resources": len(matches),
        "matched_resources_list": sorted(matches.keys()),
        "hub_matched_capacity_mw": {hub: round(cap, 3) for hub, cap in hub_cap_matched.items()},
        "hub_total_capacity_mw": {hub: round(cap, 3) for hub, cap in hub_cap_total.items()},
    }
    table["source"] = f"{table.get('source', 'defaults')}+eia923"
    return table


def normalize_hub_name(hub_name):
    if hub_name is None:
        return None
    key = str(hub_name).strip().upper()
    return HUB_ALIASES.get(key)


def infer_hub_from_coords(lat, lon):
    lat_f = _as_float(lat)
    lon_f = _as_float(lon)
    if lat_f is None or lon_f is None:
        return None

    best_hub = None
    best_dist = float("inf")
    for hub, (hub_lat, hub_lon) in HUB_COORDS.items():
        dist = ((lat_f - hub_lat) ** 2 + (lon_f - hub_lon) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_hub = hub
    return best_hub


def get_offline_threshold_mw(capacity_mw=None, pct_of_capacity=0.05, min_mw=2.0, max_mw=20.0):
    """
    Capacity-aware threshold for likely-offline interval filtering.
    Defaults to the legacy ~5 MW behavior when capacity is unavailable.
    """
    cap = _as_float(capacity_mw)
    if cap is None or cap <= 0:
        return 5.0
    threshold = cap * pct_of_capacity
    return float(max(min_mw, min(max_mw, threshold)))


def _default_table():
    return {
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "defaults",
        "model_objective": "correlation_first",
        "postprocess_config": DEFAULT_POSTPROCESS_CONFIG.copy(),
        "hub_multiplier": DEFAULT_HUB_MULTIPLIER.copy(),
        "hub_shear_alpha": DEFAULT_HUB_SHEAR_ALPHA.copy(),
        "project_multiplier": {},
        "resource_multiplier": {},
        "monthly_cf_target": _default_monthly_cf_targets(),
        "monthly_energy_target_mwh": {
            "projects": {},
            "resources": {},
            "hubs": {},
        },
        "sced_bias_hourly_multiplier": {},
        "sced_bias_month_hour_multiplier": {},
        "node_congestion_multiplier": {},
        "congestion_haircut": _deepcopy_jsonable(DEFAULT_CONGESTION_HAIRCUT),
        "reanalysis_blend": DEFAULT_REANALYSIS_BLEND.copy(),
    }


def derive_table_from_benchmark_files(repo_root=None):
    """
    Build project/hub multipliers from benchmark summary stats.
    Uses: benchmark_results_wind.json + ercot_assets.json
    """
    root = Path(repo_root) if repo_root is not None else _repo_root()
    table = _default_table()

    bench_path = root / "benchmark_results_wind.json"
    assets_path = root / "ercot_assets.json"
    if not bench_path.exists() or not assets_path.exists():
        return table

    try:
        benchmark_rows = json.loads(bench_path.read_text(encoding="utf-8"))
        assets = json.loads(assets_path.read_text(encoding="utf-8"))
    except Exception:
        return table

    project_multiplier = {}
    resource_multiplier = {}
    by_hub = defaultdict(list)

    for row in benchmark_rows:
        model_name = str(row.get("Model", ""))
        if "Advanced" not in model_name:
            continue

        project = row.get("Project")
        if not project or project not in assets:
            continue

        meta = assets[project]
        capacity = _as_float(meta.get("capacity_mw"))
        mbe = _as_float(row.get("MBE (MW)"))
        if capacity is None or capacity <= 0 or mbe is None:
            continue

        # Negative MBE means modeled < actual, so multiplier > 1.
        raw_multiplier = 1.0 - (mbe / capacity)
        multiplier = float(max(0.85, min(1.25, raw_multiplier)))
        multiplier = round(multiplier, 4)

        project_multiplier[project] = multiplier

        resource_id = meta.get("resource_name")
        if resource_id:
            resource_multiplier[str(resource_id).upper()] = multiplier

        hub_name = normalize_hub_name(meta.get("hub"))
        if hub_name:
            by_hub[hub_name].append(multiplier)

    hub_multiplier = DEFAULT_HUB_MULTIPLIER.copy()
    for hub, values in by_hub.items():
        if values:
            median_mult = statistics.median(values)
            hub_multiplier[hub] = round(float(max(0.90, min(1.20, median_mult))), 4)

    table["source"] = "derived_from_benchmark_results_wind.json"
    table["hub_multiplier"] = hub_multiplier
    table["project_multiplier"] = project_multiplier
    table["resource_multiplier"] = resource_multiplier
    return table


@lru_cache(maxsize=1)
def load_wind_calibration_table(calibration_path=None):
    path = Path(calibration_path) if calibration_path else (_repo_root() / "wind_calibration.json")
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return derive_table_from_benchmark_files(_repo_root())


def get_hub_shear_alpha(lat=None, lon=None, hub_name=None, calibration_table=None):
    table = calibration_table or load_wind_calibration_table()
    shear_map = table.get("hub_shear_alpha", {}) if isinstance(table, dict) else {}

    hub = normalize_hub_name(hub_name) or infer_hub_from_coords(lat, lon)
    if hub and hub in shear_map:
        return float(shear_map[hub]), f"hub:{hub}"

    if hub and hub in DEFAULT_HUB_SHEAR_ALPHA:
        return float(DEFAULT_HUB_SHEAR_ALPHA[hub]), f"default_hub:{hub}"

    lon_f = _as_float(lon)
    if lon_f is not None:
        # Fallback keeps old broad behavior by geography.
        return (0.22, "fallback:coastal") if lon_f > -96.0 else (0.32, "fallback:inland")

    return 0.32, "fallback:global"


def get_wind_bias_multiplier(
    lat=None,
    lon=None,
    hub_name=None,
    project_name=None,
    resource_id=None,
    calibration_table=None,
):
    table = calibration_table or load_wind_calibration_table()
    project_map = table.get("project_multiplier", {}) if isinstance(table, dict) else {}
    resource_map = table.get("resource_multiplier", {}) if isinstance(table, dict) else {}
    hub_map = table.get("hub_multiplier", {}) if isinstance(table, dict) else {}

    if project_name and project_name in project_map:
        val = _as_float(project_map.get(project_name))
        if val is not None:
            return float(max(0.85, min(1.25, val))), f"project:{project_name}"

    if resource_id:
        rid = str(resource_id).strip().upper()
        val = _as_float(resource_map.get(rid))
        if val is not None:
            return float(max(0.85, min(1.25, val))), f"resource:{rid}"

    hub = normalize_hub_name(hub_name) or infer_hub_from_coords(lat, lon)
    if hub and hub in hub_map:
        val = _as_float(hub_map.get(hub))
        if val is not None:
            return float(max(0.85, min(1.25, val))), f"hub:{hub}"

    if hub and hub in DEFAULT_HUB_MULTIPLIER:
        return float(DEFAULT_HUB_MULTIPLIER[hub]), f"default_hub:{hub}"

    return 1.0, "fallback:none"


def get_reanalysis_blend_weights(calibration_table=None):
    table = calibration_table or load_wind_calibration_table()
    blend = table.get("reanalysis_blend", {}) if isinstance(table, dict) else {}

    era5 = _as_float(blend.get("era5_weight"))
    merra2 = _as_float(blend.get("merra2_weight"))

    if era5 is None:
        era5 = DEFAULT_REANALYSIS_BLEND["era5_weight"]
    if merra2 is None:
        merra2 = DEFAULT_REANALYSIS_BLEND["merra2_weight"]

    era5 = max(0.0, float(era5))
    merra2 = max(0.0, float(merra2))
    total = era5 + merra2
    if total <= 0:
        return 1.0, 0.0
    return era5 / total, merra2 / total


def apply_monthly_wind_targets(
    gen_series,
    capacity_mw,
    hub_name=None,
    project_name=None,
    resource_id=None,
    calibration_table=None,
):
    """
    Enforces monthly wind calibration targets with this precedence:
    project monthly MWh -> resource monthly MWh -> hub monthly MWh -> hub monthly CF.
    """
    series = _safe_series(gen_series)
    if series.empty:
        return series

    cap = _as_float(capacity_mw)
    if cap is None or cap <= 0:
        return series.clip(lower=0.0)

    table = calibration_table or load_wind_calibration_table()
    hub = normalize_hub_name(hub_name)
    rid = str(resource_id).strip().upper() if resource_id else None

    monthly_energy = table.get("monthly_energy_target_mwh", {}) if isinstance(table, dict) else {}
    mwh_proj = monthly_energy.get("projects", {}) if isinstance(monthly_energy, dict) else {}
    mwh_res = monthly_energy.get("resources", {}) if isinstance(monthly_energy, dict) else {}
    mwh_hub = monthly_energy.get("hubs", {}) if isinstance(monthly_energy, dict) else {}

    mwh_target_map = None
    if project_name and project_name in mwh_proj:
        mwh_target_map = mwh_proj.get(project_name)
    elif rid and rid in mwh_res:
        mwh_target_map = mwh_res.get(rid)
    elif hub and hub in mwh_hub:
        mwh_target_map = mwh_hub.get(hub)

    cf_target_map = None
    monthly_cf = table.get("monthly_cf_target", {}) if isinstance(table, dict) else {}
    if not isinstance(monthly_cf, dict) or len(monthly_cf) == 0:
        monthly_cf = _default_monthly_cf_targets()
    if hub and isinstance(monthly_cf, dict):
        cf_target_map = monthly_cf.get(hub)

    if not isinstance(mwh_target_map, dict) and not isinstance(cf_target_map, dict):
        return series.clip(lower=0.0, upper=cap)

    interval_hours = _infer_interval_hours(series.index)
    out = series.copy()

    max_scale = _as_float(table.get("monthly_target_scale_max")) if isinstance(table, dict) else None
    min_scale = _as_float(table.get("monthly_target_scale_min")) if isinstance(table, dict) else None
    if max_scale is None:
        max_scale = 1.35
    if min_scale is None:
        min_scale = 0.70

    for month in range(1, 13):
        mask = out.index.month == month
        if not mask.any():
            continue

        month_vals = out.loc[mask]
        current_mwh = float(month_vals.sum() * interval_hours)
        if current_mwh <= 0:
            continue

        target_mwh = _lookup_month_value(mwh_target_map, month)
        if target_mwh is None:
            cf_target = _lookup_month_value(cf_target_map, month)
            if cf_target is not None:
                hours = mask.sum() * interval_hours
                target_mwh = float(cf_target) * cap * hours

        if target_mwh is None or target_mwh <= 0:
            continue

        scale = _clamp(target_mwh / current_mwh, min_scale, max_scale)
        out.loc[mask] = month_vals * scale

    return out.clip(lower=0.0, upper=cap)


def apply_sced_bias_correction(gen_series, hub_name=None, calibration_table=None):
    """
    Applies hour-of-day multipliers learned from SCED residual bias (if provided).
    """
    series = _safe_series(gen_series)
    if series.empty:
        return series

    table = calibration_table or load_wind_calibration_table()
    hub = normalize_hub_name(hub_name)
    bias_by_hub = table.get("sced_bias_hourly_multiplier", {}) if isinstance(table, dict) else {}
    mh_by_hub = table.get("sced_bias_month_hour_multiplier", {}) if isinstance(table, dict) else {}
    hour_map = bias_by_hub.get(hub, {}) if isinstance(bias_by_hub, dict) else {}
    mh_map = mh_by_hub.get(hub, {}) if isinstance(mh_by_hub, dict) else {}
    has_hour = isinstance(hour_map, dict) and len(hour_map) > 0
    has_mh = isinstance(mh_map, dict) and len(mh_map) > 0
    if not has_hour and not has_mh:
        return series

    multipliers = np.ones(len(series), dtype=float)
    hour_lookup = {str(k): _as_float(v) for k, v in hour_map.items()} if has_hour else {}
    mh_lookup = {str(k): _as_float(v) for k, v in mh_map.items()} if has_mh else {}
    for i, (month, hour) in enumerate(zip(series.index.month, series.index.hour)):
        val = mh_lookup.get(f"{int(month)}-{int(hour)}") if has_mh else None
        if val is None and has_hour:
            val = hour_lookup.get(str(int(hour)))
        if val is not None:
            multipliers[i] = _clamp(val, 0.70, 1.30)

    return pd.Series(series.values * multipliers, index=series.index, name=series.name)


def apply_node_level_adjustment(gen_series, resource_id=None, calibration_table=None):
    """
    Applies a static node-level multiplier for congestion/dispatch bias.
    """
    series = _safe_series(gen_series)
    if series.empty or not resource_id:
        return series

    table = calibration_table or load_wind_calibration_table()
    node_map = table.get("node_congestion_multiplier", {}) if isinstance(table, dict) else {}
    rid = str(resource_id).strip().upper()
    mult = _as_float(node_map.get(rid)) if isinstance(node_map, dict) else None
    if mult is None:
        return series
    mult = _clamp(mult, 0.60, 1.20)
    return pd.Series(series.values * mult, index=series.index, name=series.name)


def apply_congestion_haircut(
    gen_series,
    spp_series,
    hub_name=None,
    resource_id=None,
    calibration_table=None,
):
    """
    Applies congestion-zone curtailment haircut in low-price intervals.
    """
    series = _safe_series(gen_series)
    if series.empty:
        return series

    if spp_series is None:
        return series

    spp = _safe_series(spp_series, index=series.index).reindex(series.index).fillna(np.inf)

    table = calibration_table or load_wind_calibration_table()
    post_cfg = table.get("postprocess_config", {}) if isinstance(table, dict) else {}
    if isinstance(post_cfg, dict) and not _as_bool(post_cfg.get("apply_congestion_haircut"), default=True):
        return series

    config = table.get("congestion_haircut", {}) if isinstance(table, dict) else {}
    if not isinstance(config, dict):
        config = {}

    trigger = _as_float(config.get("price_trigger"))
    deep_trigger = _as_float(config.get("deep_price_trigger"))
    if trigger is None:
        trigger = DEFAULT_CONGESTION_HAIRCUT["price_trigger"]
    if deep_trigger is None:
        deep_trigger = DEFAULT_CONGESTION_HAIRCUT["deep_price_trigger"]

    by_hub = config.get("by_hub", {})
    deep_by_hub = config.get("deep_by_hub", {})
    resource_map = config.get("resource_multiplier", {})
    if not isinstance(by_hub, dict):
        by_hub = {}
    if not isinstance(deep_by_hub, dict):
        deep_by_hub = {}
    if not isinstance(resource_map, dict):
        resource_map = {}

    hub = normalize_hub_name(hub_name)
    base_factor = _as_float(by_hub.get(hub)) if hub else None
    deep_factor = _as_float(deep_by_hub.get(hub)) if hub else None

    if base_factor is None:
        base_factor = DEFAULT_CONGESTION_HAIRCUT["by_hub"].get(hub, 1.0)
    if deep_factor is None:
        deep_factor = DEFAULT_CONGESTION_HAIRCUT["deep_by_hub"].get(hub, base_factor)

    base_factor = _clamp(base_factor, 0.0, 1.0)
    deep_factor = _clamp(deep_factor, 0.0, 1.0)

    node_factor = 1.0
    if resource_id:
        node = _as_float(resource_map.get(str(resource_id).strip().upper()))
        if node is not None:
            node_factor = _clamp(node, 0.50, 1.20)

    multipliers = np.ones(len(series), dtype=float)
    low_mask = spp.values <= float(trigger)
    deep_mask = spp.values <= float(deep_trigger)
    multipliers[low_mask] *= base_factor
    multipliers[deep_mask] *= deep_factor
    multipliers *= node_factor

    return pd.Series(series.values * multipliers, index=series.index, name=series.name)


def apply_wind_postprocess(
    gen_series,
    capacity_mw,
    hub_name=None,
    project_name=None,
    resource_id=None,
    spp_series=None,
    calibration_table=None,
):
    """
    Applies wind-only calibration and dispatch-aware derates in a stable order.
    """
    table = calibration_table or load_wind_calibration_table()
    out = _safe_series(gen_series)
    if out.empty:
        return out

    post_cfg = table.get("postprocess_config", {}) if isinstance(table, dict) else {}
    if not isinstance(post_cfg, dict):
        post_cfg = {}
    use_monthly = _as_bool(
        post_cfg.get("apply_monthly_targets"),
        default=DEFAULT_POSTPROCESS_CONFIG["apply_monthly_targets"],
    )
    use_sced = _as_bool(
        post_cfg.get("apply_sced_bias"),
        default=DEFAULT_POSTPROCESS_CONFIG["apply_sced_bias"],
    )
    use_node = _as_bool(
        post_cfg.get("apply_node_adjustment"),
        default=DEFAULT_POSTPROCESS_CONFIG["apply_node_adjustment"],
    )
    use_haircut = _as_bool(
        post_cfg.get("apply_congestion_haircut"),
        default=DEFAULT_POSTPROCESS_CONFIG["apply_congestion_haircut"],
    )

    if use_monthly:
        out = apply_monthly_wind_targets(
            out,
            capacity_mw=capacity_mw,
            hub_name=hub_name,
            project_name=project_name,
            resource_id=resource_id,
            calibration_table=table,
        )
    if use_sced:
        out = apply_sced_bias_correction(out, hub_name=hub_name, calibration_table=table)
    if use_node:
        out = apply_node_level_adjustment(out, resource_id=resource_id, calibration_table=table)
    if use_haircut:
        out = apply_congestion_haircut(
            out,
            spp_series=spp_series,
            hub_name=hub_name,
            resource_id=resource_id,
            calibration_table=table,
        )

    cap = _as_float(capacity_mw)
    if cap is not None and cap > 0:
        out = out.clip(lower=0.0, upper=cap)
    else:
        out = out.clip(lower=0.0)
    return out


def build_and_save_calibration_table(output_path=None):
    root = _repo_root()
    table = derive_table_from_benchmark_files(root)
    table["model_objective"] = "correlation_first"
    table["postprocess_config"] = DEFAULT_POSTPROCESS_CONFIG.copy()

    assets_path = root / "ercot_assets.json"
    assets = {}
    if assets_path.exists():
        try:
            assets = json.loads(assets_path.read_text(encoding="utf-8"))
        except Exception:
            assets = {}

    eia_path, eia_year = _find_latest_eia923_generation_workbook(root)
    if assets and eia_path is not None and Path(eia_path).exists():
        try:
            table = _apply_eia923_monthly_targets(table, assets, eia_path, eia_year)
        except Exception:
            pass
    if assets:
        try:
            table = _derive_sced_month_hour_bias(table, assets, root)
        except Exception:
            pass

    out_path = Path(output_path) if output_path else (_repo_root() / "wind_calibration.json")
    out_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    return out_path
