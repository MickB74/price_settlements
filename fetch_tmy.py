import requests
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Cache directory
CACHE_DIR = "data_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_tmy_data(lat=32.4487, lon=-99.7331, force_refresh=False):
    """
    Fetches TMY data from PVGIS for the given coordinates.
    Defaults to Abilene, Texas (approximate center for West Texas wind/solar).
    """
    cache_file = os.path.join(CACHE_DIR, f"tmy_{lat}_{lon}.parquet")
    
    if not force_refresh and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            if not df.empty:
                return df
            print(f"Cache file {cache_file} is empty. Re-fetching...")
        except Exception as e:
            print(f"Error reading cache: {e}")
            
    url = "https://re.jrc.ec.europa.eu/api/tmy"
    params = {
        "lat": lat,
        "lon": lon,
        "outputformat": "json"
    }
    
    try:
        print(f"Fetching TMY data from PVGIS for {lat}, {lon}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        hourly = data['outputs']['tmy_hourly']
        df = pd.DataFrame(hourly)
        
        # Standardize columns
        # TMY has 'G(h)', 'WS10m'
        # We'll keep them as is for now, but ensure consistency with actuals
        
        df['time_str'] = df['time(UTC)']
        df['Time'] = pd.to_datetime(df['time(UTC)'], format='%Y%m%d:%H%M')
        
        # Save to cache
        df.to_parquet(cache_file)
        return df
        
    except Exception as e:
        print(f"Error fetching PVGIS TMY data: {e}")
        return pd.DataFrame()

def get_actual_data(year, lat=32.4487, lon=-99.7331, force_refresh=False):
    """
    Fetches actual hourly data for a specific year from PVGIS.
    """
    cache_file = os.path.join(CACHE_DIR, f"actual_{year}_{lat}_{lon}.parquet")
    
    if not force_refresh and os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Error reading cache: {e}")
            
    url = "https://re.jrc.ec.europa.eu/api/seriescalc"
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": year,
        "endyear": year,
        "outputformat": "json",
        "pvcalculation": 0,
        "components": 1
    }
    
    try:
        print(f"Fetching Actual data for {year} from PVGIS...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'outputs' in data and 'hourly' in data['outputs']:
            hourly = data['outputs']['hourly']
            df = pd.DataFrame(hourly)
            
            # Map columns to match TMY if needed, or handle in calculation
            # Actuals: 'Gb(i)', 'Gd(i)', 'Gr(i)', 'H_sun', 'T2m', 'WS10m', 'Int'
            # TMY: 'G(h)', 'Gb(n)', 'Gd(h)', 'IR(h)', 'WS10m', 'WD10m', 'SP'
            
            # We need Global Horizontal Irradiance.
            # seriescalc 'components=1' gives:
            # Gb(i): Beam irradiance on inclined plane (here 0 deg?) -> No, default is optimized slope?
            # Wait, we didn't specify slope/azimuth. Defaults might apply.
            # To get G(h) (Global Horizontal), we should check if it's returned.
            # The previous test showed: 'Gb(i)', 'Gd(i)', 'Gr(i)', 'H_sun', 'T2m', 'WS10m', 'Int'
            # If slope is not 0, Gb(i) is on inclined.
            # Let's try to force horizontal plane to get G(h) equivalent.
            # Or assume G(h) = Gb(i) + Gd(i) + Gr(i) if slope is 0?
            # Actually, let's request "mountingplace=free" and "angle=0" (horizontal).
            
            # Re-fetch with horizontal parameters if needed.
            # But let's check if we can just use what we have.
            # Ideally we want G(h).
            # Let's add 'angle=0' to params to ensure horizontal.
            
            # Save to cache
            df.to_parquet(cache_file)
            return df
        else:
            print(f"No hourly data found for {year}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching PVGIS Actual data: {e}")
        return pd.DataFrame()

from utils import power_curves
from utils.wind_calibration import (
    apply_wind_postprocess,
    get_hub_shear_alpha,
    get_reanalysis_blend_weights,
    get_wind_bias_multiplier,
    load_wind_calibration_table,
)
try:
    from utils.hrrr_ingestion import load_cached_hrrr_10m_wind_point
except Exception:
    def load_cached_hrrr_10m_wind_point(*args, **kwargs):
        return pd.DataFrame(), None

def solar_from_ghi(ghi_series, capacity_mw, efficiency=0.85, tracking=True, dc_ac_ratio=1.3):
    """
    Estimates solar generation from GHI.
    tracking: If True, applies a tracking gain factor (heuristic).
    dc_ac_ratio: Ratio of DC panel capacity to AC inverter capacity.
    """
    # Tracking increases yield by ~20-35% and squares off the shoulder of the curve
    if tracking:
        # Heuristic: apply a morning/evening boost by flattening the GHI curve
        effective_irradiance = ghi_series * 1.3
    else:
        effective_irradiance = ghi_series

    dc_capacity = capacity_mw * dc_ac_ratio
    dc_gen = dc_capacity * (effective_irradiance / 1000.0) * efficiency
    
    # Clip to AC Capacity (Inverter Limit)
    return dc_gen.clip(lower=0.0, upper=capacity_mw)

def wind_from_speed(speed_series, capacity_mw, turbine_type="GENERIC"):
    """Estimates wind generation from wind speed using specific power curve."""
    # Ensure numpy array for vectorization
    v = speed_series.values
    normalized_power = power_curves.get_normalized_power(v, turbine_type)
    return pd.Series(normalized_power, index=speed_series.index) * capacity_mw


def get_openmeteo_data(year, lat, lon):
    """Fetch hourly solar and wind data from Open-Meteo for any year."""
    cache_file = os.path.join(CACHE_DIR, f"openmeteo_{year}_{lat}_{lon}.parquet")
    
    # Check if cache exists (and verify columns if old cache used 100m)
    # Check if cache exists (and verify columns if old cache used 100m)
    if os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            if not df.empty and 'Wind_Speed_10m_mps' in df.columns:
                return df
            print(f"Cache file {cache_file} is empty or outdated. Re-fetching...")
        except Exception:
            pass # Ignore corrupt cache
        # If old cache (100m), ignore it and re-fetch
        
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year+1}-01-02", # Fetch TWO extra days to cover UTC-Central offset at year end and avoid boundary issues
        "hourly": "shortwave_radiation,wind_speed_10m",
        "timezone": "UTC"
    }

    # Cap end_date at today if year is current year (or future)
    today = pd.Timestamp.now().date()
    # We want to fetch up to Jan 2nd of next year if possible, but no further than today
    target_end = pd.Timestamp(f"{year+1}-01-02").date()
    safe_end = min(target_end, today)
    params["end_date"] = str(safe_end)

    
    print(f"Fetching Open-Meteo {year} data (10m) for {lat}, {lon}...")
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Open-Meteo Error: {response.text}")
            return pd.DataFrame()
            
        data = response.json()
        df = pd.DataFrame(data['hourly'])
        df['datetime'] = pd.to_datetime(df['time'])
        
        # Rename columns to standard internal names
        # shortwave_radiation = GHI (W/m2)
        # wind_speed_10m is in km/h -> Convert to m/s
        df['GHI_Wm2'] = df['shortwave_radiation']
        df['Wind_Speed_10m_mps'] = df['wind_speed_10m'] / 3.6
        
        out_df = df[['datetime', 'GHI_Wm2', 'Wind_Speed_10m_mps']].copy()
        out_df.to_parquet(cache_file)
        return out_df
        
    except Exception as e:
        print(f"Error fetching Open-Meteo data: {e}")
        return pd.DataFrame()


def get_merra2_data(year, lat, lon):
    """
    Optional local MERRA-2 cache loader.
    Expected file: data_cache/merra2_{year}_{lat}_{lon}.parquet
    """
    candidate_files = [
        os.path.join(CACHE_DIR, f"merra2_{year}_{lat}_{lon}.parquet"),
        os.path.join(CACHE_DIR, f"merra2_{year}_{round(float(lat), 4)}_{round(float(lon), 4)}.parquet"),
    ]
    cache_file = next((p for p in candidate_files if os.path.exists(p)), None)
    if not cache_file:
        return pd.DataFrame()

    try:
        df = pd.read_parquet(cache_file)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    col_map = {}
    if "datetime" in df.columns:
        col_map["datetime"] = "datetime"
    elif "time" in df.columns:
        col_map["time"] = "datetime"
    elif "Time" in df.columns:
        col_map["Time"] = "datetime"
    else:
        return pd.DataFrame()

    if "Wind_Speed_10m_mps" in df.columns:
        col_map["Wind_Speed_10m_mps"] = "Wind_Speed_10m_mps"
    elif "wind_speed_10m_mps" in df.columns:
        col_map["wind_speed_10m_mps"] = "Wind_Speed_10m_mps"
    elif "ws10m" in df.columns:
        col_map["ws10m"] = "Wind_Speed_10m_mps"
    elif "WS10M" in df.columns:
        col_map["WS10M"] = "Wind_Speed_10m_mps"
    elif "wind_speed_10m" in df.columns:
        # Some datasets store km/h. Convert conservatively.
        col_map["wind_speed_10m"] = "Wind_Speed_10m_mps"
        df = df.copy()
        df["wind_speed_10m"] = pd.to_numeric(df["wind_speed_10m"], errors="coerce") / 3.6
    else:
        return pd.DataFrame()

    out = df[list(col_map.keys())].rename(columns=col_map).copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out["Wind_Speed_10m_mps"] = pd.to_numeric(out["Wind_Speed_10m_mps"], errors="coerce")
    out = out.dropna(subset=["datetime", "Wind_Speed_10m_mps"])
    return out

# Keep old function name for backward compatibility
def get_openmeteo_2024_data(lat, lon):
    """Fetch 2024 hourly solar and wind data from Open-Meteo (backward compatibility)."""
    return get_openmeteo_data(2024, lat, lon)


WIND_MODEL_ENGINE_STANDARD = "STANDARD"
WIND_MODEL_ENGINE_ADVANCED = "ADVANCED_CALIBRATED"


def _normalize_wind_model_engine(wind_model_engine):
    key = str(wind_model_engine or WIND_MODEL_ENGINE_STANDARD).strip().upper()
    if key in {"ADVANCED", "ADVANCED_CALIBRATED", "V2", "CALIBRATED"}:
        return WIND_MODEL_ENGINE_ADVANCED
    return WIND_MODEL_ENGINE_STANDARD


def _engine_calibration_table(base_table, wind_model_engine):
    """
    Return a calibration table tuned for the selected wind engine.
    Keeps the default behavior for STANDARD and enables all advanced
    calibrations for the alternate engine.
    """
    engine_key = _normalize_wind_model_engine(wind_model_engine)
    if engine_key != WIND_MODEL_ENGINE_ADVANCED or not isinstance(base_table, dict):
        return base_table

    table = json.loads(json.dumps(base_table))
    post_cfg = table.get("postprocess_config", {})
    if not isinstance(post_cfg, dict):
        post_cfg = {}
    post_cfg["apply_monthly_targets"] = True
    post_cfg["apply_sced_bias"] = True
    post_cfg["apply_node_adjustment"] = True
    # Congestion haircut is applied downstream where SPP is available.
    post_cfg["apply_congestion_haircut"] = False
    table["postprocess_config"] = post_cfg

    # Explicit reanalysis blending for advanced mode.
    table["reanalysis_blend"] = {"era5_weight": 0.70, "merra2_weight": 0.30}
    table["advanced_engine_version"] = "wind_v2_2026_02"
    return table


def _apply_advanced_power_curve_clipping(mw_hourly, wind_speed_hub, capacity_mw):
    """
    Operational clipping logic for advanced wind mode:
    - Slightly stricter cut-in deadband.
    - Additional high-wind taper near cut-out.
    - Ramp-rate limiter to mimic fleet dispatch/availability dynamics.
    """
    out = pd.Series(pd.to_numeric(mw_hourly, errors="coerce"), index=mw_hourly.index).fillna(0.0)
    ws = pd.Series(pd.to_numeric(wind_speed_hub, errors="coerce"), index=out.index).fillna(0.0)

    out.loc[ws < 3.2] = 0.0

    mask_hi = (ws >= 22.0) & (ws < 25.0)
    if mask_hi.any():
        taper = ((25.0 - ws.loc[mask_hi]) / 3.0).clip(lower=0.0, upper=1.0)
        out.loc[mask_hi] = np.minimum(out.loc[mask_hi], taper * float(capacity_mw))

    vals = out.to_numpy(dtype=float)
    max_up = float(capacity_mw) * 0.45
    max_down = float(capacity_mw) * 0.65
    for i in range(1, len(vals)):
        vals[i] = min(vals[i], vals[i - 1] + max_up)
        vals[i] = max(vals[i], vals[i - 1] - max_down)
    return pd.Series(vals, index=out.index, name=out.name).clip(lower=0.0, upper=float(capacity_mw))

def get_profile_for_year(
    year,
    tech,
    capacity_mw,
    lat=32.4487,
    lon=-99.7331,
    force_tmy=False,
    turbine_type="GENERIC",
    hub_height=80,
    tracking=True,
    efficiency=0.85,
    hub_name=None,
    project_name=None,
    resource_id=None,
    apply_wind_calibration=False,
    turbines=None, # List of turbine dicts for blended profile
    wind_weather_source="AUTO",
    hrrr_forecast_hour=0,
    wind_model_engine=WIND_MODEL_ENGINE_STANDARD,
):
    """
    Generates a full year profile.
    Uses Actual data for 2005-2023 (PVGIS).
    Uses Open-Meteo for 2024+ (Solar + Wind) - real weather data.
    Uses TMY data only when force_tmy=True.
    efficiency: System efficiency (default 0.85 for 15% losses). Pass 0.86 for 14% losses.
    """
    
    wind_model_engine_key = _normalize_wind_model_engine(wind_model_engine)

    calibration_table = None
    if tech == "Wind" and apply_wind_calibration:
        calibration_table = load_wind_calibration_table()
        calibration_table = _engine_calibration_table(calibration_table, wind_model_engine_key)

    # Handle Blended Profile
    if turbines and len(turbines) > 0 and tech == "Wind":
        blended_series = get_blended_profile_for_year(
            year=year,
            tech=tech,
            turbines=turbines,
            lat=lat,
            lon=lon,
            hub_height=hub_height,
            efficiency=efficiency,
            wind_weather_source=wind_weather_source,
            hrrr_forecast_hour=hrrr_forecast_hour,
            wind_model_engine=wind_model_engine_key,
        )
        if apply_wind_calibration and blended_series is not None and not blended_series.empty:
            blended_series = apply_wind_postprocess(
                blended_series,
                capacity_mw=capacity_mw,
                hub_name=hub_name,
                project_name=project_name,
                resource_id=resource_id,
                calibration_table=calibration_table,
            )
        return blended_series
    
    # PRIORITY 1: Check for pregenerated profiles (committed to repo for Streamlit Cloud)
    # This eliminates API dependency issues
    PREGEN_DIR = os.path.join(CACHE_DIR, "pregenerated")
    if os.path.exists(PREGEN_DIR) and not force_tmy:
        # Try to find matching pregenerated profile by location proximity
        HUB_LOCATIONS = {
            "HB_NORTH": (32.3865, -96.8475),
            "HB_SOUTH": (26.9070, -99.2715),
            "HB_WEST": (32.4518, -100.5371),
            "HB_HOUSTON": (29.3013, -94.7977),
            "HB_PAN": (35.2220, -101.8313),
        }
        
        # Find closest hub (within 0.1 degrees ~= 11 km)
        for hub_name, (hub_lat, hub_lon) in HUB_LOCATIONS.items():
            if abs(lat - hub_lat) < 0.1 and abs(lon - hub_lon) < 0.1:
                pregen_file = os.path.join(PREGEN_DIR, f"{hub_name}_{tech}_{year}.parquet")
                if os.path.exists(pregen_file):
                    try:
                        df = pd.read_parquet(pregen_file)
                        if not df.empty and len(df) > 30000:
                            # Scale to requested capacity
                            series = pd.Series(
                                df['gen_mw'].values * (capacity_mw / 100.0),
                                index=pd.to_datetime(df['datetime'], utc=True),
                                name="Gen_MW"
                            )
                            if tech == "Wind" and apply_wind_calibration:
                                bias_mult, _ = get_wind_bias_multiplier(
                                    lat=lat,
                                    lon=lon,
                                    hub_name=hub_name,
                                    project_name=project_name,
                                    resource_id=resource_id,
                                    calibration_table=calibration_table,
                                )
                                series = (series * bias_mult).clip(lower=0.0, upper=capacity_mw)
                                series = apply_wind_postprocess(
                                    series,
                                    capacity_mw=capacity_mw,
                                    hub_name=hub_name,
                                    project_name=project_name,
                                    resource_id=resource_id,
                                    calibration_table=calibration_table,
                                )
                            print(f"✓ Using pregenerated profile: {hub_name}_{tech}_{year}")
                            return series
                    except Exception as e:
                        print(f"Warning: Could not load pregenerated profile: {e}")
                break
    
    # PRIORITY 2-4: API sources (PVGIS, OpenMeteo, TMY) - original logic
    # Determine Data Source
    # If forced TMY, disable all actuals
    use_pvgis_actual = (2005 <= year <= 2023) and not force_tmy
    use_openmeteo_actual = (year >= 2024) and not force_tmy  # 2024 and all future years
    wind_weather_source_key = str(wind_weather_source or "AUTO").strip().upper()
    
    df_data = pd.DataFrame()
    source_type = "TMY" # Default

    # 0. Optional HRRR cache path for Wind (separate from default API path).
    if tech == "Wind" and not force_tmy and wind_weather_source_key == "NOAA_HRRR_CACHED":
        try:
            hrrr_df, hrrr_path = load_cached_hrrr_10m_wind_point(
                start_time=f"{year}-01-01T00:00:00Z",
                end_time=f"{year}-12-31T23:00:00Z",
                lat=lat,
                lon=lon,
                forecast_hour=int(hrrr_forecast_hour),
                model="hrrr",
                product="sfc",
            )
            if not hrrr_df.empty:
                df_data = pd.DataFrame(
                    {
                        "datetime": pd.to_datetime(hrrr_df["valid_time_utc"], utc=True, errors="coerce"),
                        "Wind_Speed_10m_mps": pd.to_numeric(hrrr_df["wind_speed_10m_mps"], errors="coerce"),
                    }
                ).dropna(subset=["datetime", "Wind_Speed_10m_mps"])
                source_type = "HRRR_Cached"
                print(
                    f"✓ HRRR cache loaded for {year} "
                    f"({len(df_data)} rows, f{int(hrrr_forecast_hour):02d}) "
                    f"from {hrrr_path}"
                )
            else:
                print(
                    f"HRRR cache not found/empty for {year} at ({lat:.4f}, {lon:.4f}) "
                    f"f{int(hrrr_forecast_hour):02d}; falling back to default weather source."
                )
        except Exception as e:
            print(f"HRRR cache load failed for {year}: {e}. Falling back to default weather source.")
    
    # 1. Try PVGIS Actuals (2005-2023)
    if use_pvgis_actual and df_data.empty:
        try:
            cache_file = os.path.join(CACHE_DIR, f"actual_{year}_{lat}_{lon}.parquet")
            if os.path.exists(cache_file):
                 try:
                     cached_df = pd.read_parquet(cache_file)
                     if not cached_df.empty:
                         df_data = cached_df
                     else:
                         print(f"Cache file {cache_file} is empty. Ignoring.")
                 except Exception:
                     pass # Ignore corrupt cache
            
            if df_data.empty: # Only fetch if cache miss or empty cache
                url = "https://re.jrc.ec.europa.eu/api/seriescalc"
                params = {
                    "lat": lat, "lon": lon, "startyear": year, "endyear": year,
                    "outputformat": "json", "pvcalculation": 0, "components": 1,
                    "angle": 0 # Horizontal
                }
                print(f"Fetching PVGIS Actual data for {year} from {url} with params {params}...")
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if 'outputs' in data and 'hourly' in data['outputs']:
                        df_data = pd.DataFrame(data['outputs']['hourly'])
                        df_data.to_parquet(cache_file)
                    else:
                        raise ValueError(f"PVGIS returned 200 but no 'hourly' data found. Response keys: {list(data.keys())}")
                else:
                    raise  ValueError(f"PVGIS API Error: {response.status_code} - {response.text[:200]}")
            
            if not df_data.empty:
                source_type = "Actual"
        except Exception as e:
            print(f"Error in PVGIS flow: {e}")
            if use_openmeteo_actual: # If we can fallback to OpenMeteo, log warning but continue
                print("Falling back to next source...")
            else:
                 # If this was our primary source and we can't fallback easily (or we want to know why it failed), re-raise
                 # Actually, we should let the fallback logic (TMY) try. 
                 # But if TMY also fails, we want to know the FIRST error.
                 # Let's attach the error to the empty dataframe? No.
                 # Let's print it to stdout (which is captured in debug log) AND proceed.
                 pass

    # 2. Try Open-Meteo (2024+ OR Fallback for PVGIS failure)
    # CRITICAL FIX: If PVGIS failed for 2005-2023, use Open-Meteo as robust fallback
    if df_data.empty and (use_openmeteo_actual or (not force_tmy)):
        print(f"Fetching Open-Meteo data for {year} ({'primary' if use_openmeteo_actual else 'PVGIS fallback'})...")
        df_om = get_openmeteo_data(year, lat, lon)
        if not df_om.empty:
            df_data = df_om
            source_type = "OpenMeteo_Actual"
            print(f"✓ Open-Meteo data loaded for {year}")

    # 3. Fallback to TMY (or if enabled)
    if df_data.empty:
        df_data = get_tmy_data(lat, lon)
        source_type = "TMY"
    
    if df_data.empty:
        return pd.Series()

    # Calculate MW from Weather Data
    if tech == "Solar":
        if source_type == "Actual":
            # PVGIS: G(h) = Gb(i) + Gd(i) (approx on horizontal)
            irradiance = df_data['Gb(i)'] + df_data['Gd(i)'] + df_data.get('Gr(i)', 0)
        elif source_type == "OpenMeteo_Actual":
            # Open-Meteo: GHI provided directly
            irradiance = df_data['GHI_Wm2']
        elif source_type == "TMY":
            # PVGIS TMY: G(h)
            irradiance = df_data['G(h)']
        else:
            return pd.Series()
            
        mw_hourly = solar_from_ghi(irradiance, capacity_mw, tracking=tracking, efficiency=efficiency)
        
    elif tech == "Wind":
        if source_type in ["Actual", "TMY", "OpenMeteo_Actual", "HRRR_Cached"]:
            if 'WS10m' in df_data.columns or 'Wind_Speed_10m_mps' in df_data.columns:
                # Use 10m wind speed and apply scaling factor
                if 'WS10m' in df_data.columns:
                    ws_10m = df_data['WS10m']
                else:
                    ws_10m = df_data['Wind_Speed_10m_mps']

                if (
                    source_type in ["OpenMeteo_Actual", "HRRR_Cached"]
                    and apply_wind_calibration
                    and calibration_table is not None
                ):
                    era5_w, merra2_w = get_reanalysis_blend_weights(calibration_table=calibration_table)
                    if merra2_w > 0:
                        df_merra = get_merra2_data(year, lat, lon)
                        if not df_merra.empty:
                            era5_df = pd.DataFrame(
                                {
                                    "datetime": pd.to_datetime(df_data["datetime"], utc=True),
                                    "era5_ws10m": pd.to_numeric(ws_10m, errors="coerce"),
                                }
                            )
                            merra_df = df_merra.rename(columns={"Wind_Speed_10m_mps": "merra_ws10m"}).copy()
                            merged = era5_df.merge(merra_df, on="datetime", how="left")
                            merged["merra_ws10m"] = merged["merra_ws10m"].interpolate().ffill().bfill()
                            ws_10m = (
                                merged["era5_ws10m"] * era5_w
                                + merged["merra_ws10m"] * merra2_w
                            )
                
                # Hub-aware scaling at hub height (power law: v_h = v_10 * (h/10)^alpha).
                alpha, _ = get_hub_shear_alpha(
                    lat=lat,
                    lon=lon,
                    hub_name=hub_name,
                    calibration_table=calibration_table,
                )
                wind_speed_hub = ws_10m * ((hub_height / 10.0) ** alpha)
                mw_hourly = wind_from_speed(wind_speed_hub, capacity_mw, turbine_type=turbine_type) * efficiency
                if apply_wind_calibration:
                    bias_mult, _ = get_wind_bias_multiplier(
                        lat=lat,
                        lon=lon,
                        hub_name=hub_name,
                        project_name=project_name,
                        resource_id=resource_id,
                        calibration_table=calibration_table,
                    )
                    mw_hourly = (mw_hourly * bias_mult).clip(lower=0.0, upper=capacity_mw)
                if wind_model_engine_key == WIND_MODEL_ENGINE_ADVANCED:
                    mw_hourly = _apply_advanced_power_curve_clipping(
                        mw_hourly=mw_hourly,
                        wind_speed_hub=wind_speed_hub,
                        capacity_mw=capacity_mw,
                    )
            else:
                return pd.Series()
        else:
            return pd.Series()
    else:
        return pd.Series()

    # Align with Target Year (Resampling)
    if source_type in ["Actual", "OpenMeteo_Actual", "HRRR_Cached"]:
        # Parse timestamps
        if source_type in ["OpenMeteo_Actual", "HRRR_Cached"]:
            timestamps = pd.to_datetime(df_data['datetime'], utc=True, errors='coerce')
            if timestamps.dt.tz is None:
                timestamps = timestamps.dt.tz_localize('UTC')
            else:
                timestamps = timestamps.dt.tz_convert('UTC')
        else:
            # PVGIS format
            if 'time' in df_data.columns: time_col = 'time'
            elif 'time(UTC)' in df_data.columns: time_col = 'time(UTC)'
            else: return pd.Series()
            timestamps = pd.to_datetime(df_data[time_col], format='%Y%m%d:%H%M', utc=True)
            
            # --- FIX: handle 30 minute delay for PVGIS ---
            # PVGIS hourly data is often "time ending" or integrated.
            # User reported 30-min lag. Shift back by 30 mins to align center of mass?
            # Or if it's hour-ending (10:00 is 9-10), center is 9:30.
            # Shifting -30 mins aligns it better with instantaneous expectations.
            timestamps = timestamps - pd.Timedelta(minutes=30)
        
        # Create Series
        s_hourly = pd.Series(mw_hourly.values, index=timestamps)
        
        # Handle duplicates/Resample
        s_hourly = s_hourly[~s_hourly.index.duplicated(keep='first')]
        if source_type == "HRRR_Cached":
            # Keep HRRR gaps explicit so sparse cache windows do not linearly smear across months.
            s_15min = s_hourly.resample('15min').asfreq()
            s_15min = s_15min.interpolate(method='time', limit=4, limit_direction='both')
        else:
            s_15min = s_hourly.resample('15min').interpolate(method='linear')
        
        # Reindex to full year (aligned to US/Central)
        target_index_cst = pd.date_range(
            start=f"{year}-01-01 00:00", 
            end=f"{year}-12-31 23:45", 
            freq='15min', 
            tz='US/Central'
        )
        target_index = target_index_cst.tz_convert('UTC')
        
        aligned = s_15min.reindex(target_index)
        hrrr_coverage = None
        if source_type == "HRRR_Cached":
            # Avoid spreading short HRRR windows across a full year.
            hrrr_coverage = float(aligned.notna().mean()) if len(aligned) else 0.0
            if hrrr_coverage >= 0.95:
                s_final = aligned.ffill().bfill()
            else:
                s_final = aligned.interpolate(method='time', limit=4, limit_direction='both')
                print(
                    f"HRRR cache coverage for {year} is {hrrr_coverage:.1%}; "
                    "missing intervals will be backfilled from default weather."
                )
        else:
            s_final = aligned.ffill().bfill()
        s_final.name = "Gen_MW"
        if tech == "Wind" and apply_wind_calibration:
            s_final = apply_wind_postprocess(
                s_final,
                capacity_mw=capacity_mw,
                hub_name=hub_name,
                project_name=project_name,
                resource_id=resource_id,
                calibration_table=calibration_table,
            )
        if (
            source_type == "HRRR_Cached"
            and hrrr_coverage is not None
            and hrrr_coverage < 0.95
        ):
            try:
                fallback_series = get_profile_for_year(
                    year=year,
                    tech=tech,
                    capacity_mw=capacity_mw,
                    lat=lat,
                    lon=lon,
                    force_tmy=force_tmy,
                    turbine_type=turbine_type,
                    hub_height=hub_height,
                    tracking=tracking,
                    efficiency=efficiency,
                    hub_name=hub_name,
                    project_name=project_name,
                    resource_id=resource_id,
                    apply_wind_calibration=apply_wind_calibration,
                    turbines=None,
                    wind_weather_source="AUTO",
                    hrrr_forecast_hour=hrrr_forecast_hour,
                    wind_model_engine=wind_model_engine_key,
                )
                if not fallback_series.empty:
                    if fallback_series.index.tz is None:
                        fallback_series.index = fallback_series.index.tz_localize("UTC")
                    else:
                        fallback_series.index = fallback_series.index.tz_convert("UTC")
                    fallback_series = fallback_series.reindex(s_final.index)
                    # Blend HRRR with default source where both exist to avoid severe
                    # downward shifts from source-specific low wind bias.
                    overlap_mask = s_final.notna() & fallback_series.notna()
                    if overlap_mask.any():
                        hrrr_weight = 0.70
                        s_final.loc[overlap_mask] = (
                            s_final.loc[overlap_mask] * hrrr_weight
                            + fallback_series.loc[overlap_mask] * (1.0 - hrrr_weight)
                        )
                        print(
                            f"HRRR/default blend applied for {year} on "
                            f"{int(overlap_mask.sum()):,} intervals (HRRR weight={hrrr_weight:.2f})."
                        )
                    missing_before = int(s_final.isna().sum())
                    s_final = s_final.combine_first(fallback_series)
                    missing_after = int(s_final.isna().sum())
                    print(
                        f"HRRR backfill applied for {year}: "
                        f"{missing_before - missing_after:,} intervals filled."
                    )
            except Exception as e:
                print(f"HRRR backfill fallback failed for {year}: {e}")
        return s_final.fillna(0)

        
    else:
        # TMY Logic (TMY from PVGIS has UTC timestamps)
        # --- FIX: Fix 6-hour Timezone Shift ---
        # Instead of dummy index, we keep the UTC time info but swap the year to target year.
        
        # timestamps are usually in year matches source (e.g. 2005-2020 mix)
        # PVGIS TMY returns a JSON with 'time(UTC)' that includes a year.
        # We parse it above in get_tmy_data line 46: df['Time']
        
        # --- FIX: Normalize to single dummy year (2000) BEFORE resampling ---
        # TMY data pulls Jan from e.g. 2009, Feb from 2014.
        # If we rely on original timestamps, resample() will creating massive gaps 
        # (years of NaNs) and interpolate across them. 
        # Normalizing to 2000 forces them to be contiguous Jan->Dec.
        
        tmy_times_raw = pd.to_datetime(df_data['time(UTC)'], format='%Y%m%d:%H%M', utc=True) 
        
        # Use year 2000 (Leap Year) to safely handle any Feb 29s if present, 
        # and allow interpolation for Feb 29 if missing.
        tmy_times_norm = tmy_times_raw.map(lambda t: t.replace(year=2000))
        
        # Create Series with Normalized Index
        s_hourly = pd.Series(mw_hourly.values, index=tmy_times_norm)
        
        # Sort (should be already sorted, but safe)
        s_hourly = s_hourly.sort_index()
        
        # Resample to 15-min (interpolates strictly between contiguous hours)
        s_15min_source = s_hourly.resample('15min').interpolate(method='linear')
        
        # Create 'key' column
        df_source = pd.DataFrame({'mw': s_15min_source.values, 'time': s_15min_source.index})
        # Extract M-D-H-M from Source
        df_source['key'] = df_source['time'].dt.strftime('%m-%d-%H-%M')
        # Drop duplicates in key 
        df_source = df_source.drop_duplicates(subset=['key'])
        
        # Create Target Index (UTC)
        target_index_cst = pd.date_range(
            start=f"{year}-01-01 00:00", 
            end=f"{year}-12-31 23:45", 
            freq='15min', 
            tz='US/Central'
        )
        target_index_utc = target_index_cst.tz_convert('UTC')
        
        # Create Target Frame
        df_target = pd.DataFrame(index=target_index_utc)
        df_target['key'] = df_target.index.strftime('%m-%d-%H-%M')
        
        # Join
        df_merged = df_target.merge(df_source[['key', 'mw']], on='key', how='left')
        
        # Fill missing (e.g. Leap day diffs, or Feb 29 handling)
        df_merged['mw'] = df_merged['mw'].interpolate(method='linear').ffill().bfill()
        
        out_series = pd.Series(df_merged['mw'].values, index=target_index_utc, name="Gen_MW")
        if tech == "Wind" and apply_wind_calibration:
            out_series = apply_wind_postprocess(
                out_series,
                capacity_mw=capacity_mw,
                hub_name=hub_name,
                project_name=project_name,
                resource_id=resource_id,
                calibration_table=calibration_table,
            )
        return out_series


def get_blended_profile_for_year(
    year,
    tech,
    turbines, # List of dicts: [{'type': 'NORDEX_N149', 'count': 65, 'capacity_mw': 4.5}, ...]
    lat=32.4487,
    lon=-99.7331,
    hub_height=80, # Default if not specified in turbine dict
    efficiency=0.85,
    wind_weather_source="AUTO",
    hrrr_forecast_hour=0,
    wind_model_engine=WIND_MODEL_ENGINE_STANDARD,
):
    """
    Generates a blended profile for a project with multiple turbine types.
    """
    total_mw = sum(t['count'] * t['capacity_mw'] for t in turbines)
    if total_mw == 0:
        return pd.Series()
        
    blended_series = None
    
    for t in turbines:
        t_type = t.get('type', "GENERIC")
        count = t.get('count', 1)
        cap_per_turbine = t.get('capacity_mw', 2.0)
        sub_total_cap = count * cap_per_turbine
        t_height = t.get('hub_height_m', hub_height)
        
        # Get profile for this sub-group
        print(f"Generating sub-profile: {count}x {t_type} ({sub_total_cap:.1f} MW)")
        s = get_profile_for_year(
            year=year,
            tech=tech,
            capacity_mw=sub_total_cap,
            lat=lat,
            lon=lon,
            hub_height=t_height,
            turbine_type=t_type,
            efficiency=efficiency,
            force_tmy=False,
            wind_weather_source=wind_weather_source,
            hrrr_forecast_hour=hrrr_forecast_hour,
            wind_model_engine=wind_model_engine,
        )
        
        if s.empty:
            continue
            
        if blended_series is None:
            blended_series = s
        else:
            blended_series = blended_series.add(s, fill_value=0)
            
    if blended_series is None:
        return pd.Series()
        
    blended_series.name = "Gen_MW"
    return blended_series


if __name__ == "__main__":
    print("Testing fetch_tmy (Hybrid + Open-Meteo + Forced TMY)...")
    
    # Test Forced TMY 2024
    print("\n--- Test 2024 Solar (Forced TMY) ---")
    s_2024_tmy = get_profile_for_year(2024, "Solar", 100, force_tmy=True)
    print(f"Solar 2024 (TMY): {len(s_2024_tmy)} points, Max: {s_2024_tmy.max():.2f} MW")
    
    # Check for flatline
    head_vals = s_2024_tmy.head(20).tolist()
    print(f"Head values: {head_vals}")
    if len(set(head_vals)) == 1:
        print("⚠️ WARNING: Flatline detected at start of year!")
