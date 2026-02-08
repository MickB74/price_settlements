import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

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


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _as_float(value):
    try:
        out = float(value)
        return out
    except (TypeError, ValueError):
        return None


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
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "defaults",
        "hub_multiplier": DEFAULT_HUB_MULTIPLIER.copy(),
        "hub_shear_alpha": DEFAULT_HUB_SHEAR_ALPHA.copy(),
        "project_multiplier": {},
        "resource_multiplier": {},
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


def build_and_save_calibration_table(output_path=None):
    table = derive_table_from_benchmark_files(_repo_root())
    out_path = Path(output_path) if output_path else (_repo_root() / "wind_calibration.json")
    out_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    return out_path
