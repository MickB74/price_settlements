"""
Standalone HRRR ingestion utilities.

This module is intentionally separate from the default forecast path.
It fetches point-level 10m wind from HRRR and caches to parquet for
later experimentation/comparison.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


HUB_LOCATIONS = {
    "HB_NORTH": (32.3865, -96.8475),
    "HB_SOUTH": (26.9070, -99.2715),
    "HB_WEST": (32.4518, -100.5371),
    "HB_HOUSTON": (29.3013, -94.7977),
    "HB_PAN": (35.2220, -101.8313),
}


@dataclass(frozen=True)
class HRRRIngestionConfig:
    model: str = "hrrr"
    product: str = "sfc"
    forecast_hour: int = 0
    cycle_hours: int = 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _hrrr_cache_root(repo_root: Optional[Path] = None) -> Path:
    root = repo_root if repo_root is not None else _repo_root()
    path = root / "data_cache" / "hrrr"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _iter_cycles(start_utc: pd.Timestamp, end_utc: pd.Timestamp, cycle_hours: int) -> Iterable[pd.Timestamp]:
    step = pd.Timedelta(hours=max(1, int(cycle_hours)))
    cur = start_utc.floor("h")
    stop = end_utc.floor("h")
    while cur <= stop:
        yield cur
        cur += step


def _load_herbie_class():
    try:
        from herbie import Herbie  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "HRRR ingestion requires optional packages. Install with: "
            "`python -m pip install herbie cfgrib eccodes`"
        ) from exc
    return Herbie


def _find_coord_name(dataset, candidates):
    existing = list(dataset.coords) + list(dataset.dims)
    lower = {str(k).lower(): str(k) for k in existing}
    for cand in candidates:
        key = lower.get(cand.lower())
        if key:
            return key
    for key in existing:
        lkey = str(key).lower()
        if "lat" in lkey:
            if "latitude" in candidates:
                return str(key)
        if "lon" in lkey:
            if "longitude" in candidates:
                return str(key)
    return None


def _find_wind_var_name(dataset, component: str):
    component = component.lower()
    names = list(dataset.data_vars)
    lower = {str(k).lower(): str(k) for k in names}

    preferred = [
        f"{component}10",
        f"{component}grd10m",
        f"{component}_10maboveground",
        f"{component}-component_of_wind_height_above_ground",
        f"{component}grd",
    ]
    for cand in preferred:
        key = lower.get(cand)
        if key:
            return key

    for name in names:
        lname = str(name).lower()
        if component in lname and "10" in lname and ("ground" in lname or "m" in lname):
            return str(name)
    return None


def _cache_file_path(
    lat: float,
    lon: float,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    config: HRRRIngestionConfig,
    cache_root: Path,
) -> Path:
    payload = {
        "lat": round(float(lat), 5),
        "lon": round(float(lon), 5),
        "start": start_utc.isoformat(),
        "end": end_utc.isoformat(),
        "model": config.model,
        "product": config.product,
        "forecast_hour": int(config.forecast_hour),
        "cycle_hours": int(config.cycle_hours),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    fname = (
        f"hrrr_{config.model}_{config.product}_f{int(config.forecast_hour):02d}_"
        f"{start_utc.strftime('%Y%m%d%H')}_{end_utc.strftime('%Y%m%d%H')}_{digest}.parquet"
    )
    return cache_root / fname


def _extract_point_from_dataset(dataset, lat: float, lon: float):
    lat_name = _find_coord_name(dataset, ["latitude", "lat"])
    lon_name = _find_coord_name(dataset, ["longitude", "lon"])
    if not lat_name or not lon_name:
        raise ValueError("Unable to locate latitude/longitude coordinates in HRRR dataset.")

    u_name = _find_wind_var_name(dataset, "u")
    v_name = _find_wind_var_name(dataset, "v")
    if not u_name or not v_name:
        raise ValueError("Unable to locate 10m U/V wind variables in HRRR dataset.")

    point = dataset.sel({lat_name: float(lat), lon_name: float(lon)}, method="nearest")
    u10 = float(np.asarray(point[u_name]).squeeze())
    v10 = float(np.asarray(point[v_name]).squeeze())
    grid_lat = float(np.asarray(point[lat_name]).squeeze())
    grid_lon = float(np.asarray(point[lon_name]).squeeze())
    return u10, v10, grid_lat, grid_lon


def fetch_hrrr_10m_wind_point(
    start_time,
    end_time,
    lat: float,
    lon: float,
    config: Optional[HRRRIngestionConfig] = None,
    force_refresh: bool = False,
    repo_root: Optional[Path] = None,
):
    """
    Fetch HRRR 10m wind for a single point and return a cached DataFrame.

    Returns dataframe columns:
    - valid_time_utc
    - model_cycle_utc
    - forecast_hour
    - u10_mps
    - v10_mps
    - wind_speed_10m_mps
    - grid_lat
    - grid_lon
    - source
    - grib_path
    """
    cfg = config or HRRRIngestionConfig()
    start_utc = _to_utc_timestamp(start_time)
    end_utc = _to_utc_timestamp(end_time)
    if end_utc < start_utc:
        raise ValueError("end_time must be >= start_time")

    cache_root = _hrrr_cache_root(repo_root=repo_root)
    cache_path = _cache_file_path(lat, lon, start_utc, end_utc, cfg, cache_root)
    meta_path = cache_path.with_suffix(".json")
    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        return cached, cache_path

    Herbie = _load_herbie_class()
    records = []
    failures = []

    for cycle in _iter_cycles(start_utc, end_utc, cfg.cycle_hours):
        try:
            model_run = Herbie(
                cycle.to_pydatetime(),
                model=cfg.model,
                product=cfg.product,
                fxx=int(cfg.forecast_hour),
            )
            dataset = model_run.xarray(":(UGRD|VGRD):10 m above ground:")
            u10, v10, grid_lat, grid_lon = _extract_point_from_dataset(dataset, lat, lon)
            speed = float(np.sqrt(u10 * u10 + v10 * v10))
            records.append(
                {
                    "valid_time_utc": cycle + pd.Timedelta(hours=int(cfg.forecast_hour)),
                    "model_cycle_utc": cycle,
                    "forecast_hour": int(cfg.forecast_hour),
                    "u10_mps": u10,
                    "v10_mps": v10,
                    "wind_speed_10m_mps": speed,
                    "grid_lat": grid_lat,
                    "grid_lon": grid_lon,
                    "source": str(getattr(model_run, "grib_source", "")),
                    "grib_path": str(getattr(model_run, "grib", "")),
                }
            )
        except Exception as exc:
            failures.append({"cycle_utc": cycle.isoformat(), "error": str(exc)})

    if not records:
        raise RuntimeError(
            "HRRR ingestion produced no records. "
            "If dependencies are installed, verify upstream HRRR availability."
        )

    out = pd.DataFrame(records).sort_values("valid_time_utc").reset_index(drop=True)
    out.to_parquet(cache_path, index=False)

    meta = {
        "lat": float(lat),
        "lon": float(lon),
        "start_utc": start_utc.isoformat(),
        "end_utc": end_utc.isoformat(),
        "config": {
            "model": cfg.model,
            "product": cfg.product,
            "forecast_hour": int(cfg.forecast_hour),
            "cycle_hours": int(cfg.cycle_hours),
        },
        "rows": int(len(out)),
        "failures": failures[:200],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, cache_path


def resolve_point_from_selector(
    hub: Optional[str] = None,
    project: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    assets_path: Optional[Path] = None,
):
    """
    Resolve a point from hub/project/lat+lon selectors.
    """
    if lat is not None and lon is not None:
        return float(lat), float(lon), f"point_{float(lat):.4f}_{float(lon):.4f}"

    if hub:
        hub_key = str(hub).strip().upper()
        if hub_key not in HUB_LOCATIONS:
            raise ValueError(f"Unknown hub '{hub}'. Supported: {sorted(HUB_LOCATIONS.keys())}")
        h_lat, h_lon = HUB_LOCATIONS[hub_key]
        return float(h_lat), float(h_lon), hub_key

    if project:
        root = _repo_root()
        path = assets_path or (root / "ercot_assets.json")
        if not path.exists():
            raise FileNotFoundError(f"Assets file not found: {path}")
        assets = json.loads(path.read_text(encoding="utf-8"))
        if project not in assets:
            raise ValueError(f"Project not found in assets: '{project}'")
        meta = assets[project]
        p_lat = meta.get("lat")
        p_lon = meta.get("lon")
        if p_lat is None or p_lon is None:
            raise ValueError(f"Project '{project}' is missing lat/lon in assets.")
        return float(p_lat), float(p_lon), str(project)

    raise ValueError("Provide either hub, project, or lat+lon.")


def load_cached_hrrr_10m_wind_point(
    start_time,
    end_time,
    lat: float,
    lon: float,
    forecast_hour: int = 0,
    model: str = "hrrr",
    product: str = "sfc",
    repo_root: Optional[Path] = None,
    max_distance_deg: float = 0.30,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """
    Load cached HRRR 10m wind for a point/time range without fetching from network.

    Returns:
    - DataFrame with at least ['valid_time_utc', 'wind_speed_10m_mps'] when found
    - One representative cache path used (or None)
    """
    start_utc = _to_utc_timestamp(start_time)
    end_utc = _to_utc_timestamp(end_time)
    if end_utc < start_utc:
        raise ValueError("end_time must be >= start_time")

    cache_root = _hrrr_cache_root(repo_root=repo_root)
    if not cache_root.exists():
        return pd.DataFrame(), None

    lat = float(lat)
    lon = float(lon)
    candidates = []

    for meta_path in cache_root.glob("*.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        cfg = meta.get("config", {})
        if str(cfg.get("model", "")).lower() != str(model).lower():
            continue
        if str(cfg.get("product", "")).lower() != str(product).lower():
            continue
        if int(cfg.get("forecast_hour", -1)) != int(forecast_hour):
            continue

        try:
            m_lat = float(meta.get("lat"))
            m_lon = float(meta.get("lon"))
        except Exception:
            continue
        dist = float(((m_lat - lat) ** 2 + (m_lon - lon) ** 2) ** 0.5)
        if dist > float(max_distance_deg):
            continue

        try:
            c_start = _to_utc_timestamp(meta.get("start_utc"))
            c_end = _to_utc_timestamp(meta.get("end_utc"))
        except Exception:
            continue
        if c_end < start_utc or c_start > end_utc:
            continue

        parquet_path = meta_path.with_suffix(".parquet")
        if not parquet_path.exists():
            continue

        overlap_start = max(start_utc, c_start)
        overlap_end = min(end_utc, c_end)
        overlap_hours = max(0.0, (overlap_end - overlap_start).total_seconds() / 3600.0)
        candidates.append((dist, -overlap_hours, c_start, parquet_path))

    if not candidates:
        return pd.DataFrame(), None

    candidates.sort()
    frames = []
    used_path = None
    for _, _, _, parquet_path in candidates:
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            continue
        if df.empty or "valid_time_utc" not in df.columns or "wind_speed_10m_mps" not in df.columns:
            continue

        local = df.copy()
        local["valid_time_utc"] = pd.to_datetime(local["valid_time_utc"], utc=True, errors="coerce")
        local = local.dropna(subset=["valid_time_utc", "wind_speed_10m_mps"])
        local = local[
            (local["valid_time_utc"] >= start_utc) &
            (local["valid_time_utc"] <= end_utc)
        ].copy()
        if local.empty:
            continue

        frames.append(local)
        if used_path is None:
            used_path = parquet_path

    if not frames:
        return pd.DataFrame(), None

    out = pd.concat(frames, ignore_index=True)
    if "model_cycle_utc" in out.columns:
        out["model_cycle_utc"] = pd.to_datetime(out["model_cycle_utc"], utc=True, errors="coerce")
        out = out.sort_values(["valid_time_utc", "model_cycle_utc"])
    else:
        out = out.sort_values("valid_time_utc")
    out = out.drop_duplicates(subset=["valid_time_utc"], keep="last").reset_index(drop=True)
    return out, used_path
