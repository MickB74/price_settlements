"""
Benchmark comparison: Current heuristic solar model vs PVLib physics model.

For each solar project, generates profiles using both approaches
from the same GHI weather data (Open-Meteo ERA5), then compares
against ERCOT SCED actual generation.
"""

import pandas as pd
import numpy as np
import json
import os

import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

import fetch_tmy
import sced_fetcher


def solar_pvlib_from_ghi(
    ghi_series,
    timestamps_utc,
    lat,
    lon,
    capacity_mw,
    tracking=True,
    dc_ac_ratio=1.3,
    tilt=None,
):
    """
    PVLib-based solar generation model.

    Uses GHI decomposition, solar position, and either single-axis tracking
    or fixed-tilt geometry to compute plane-of-array irradiance, then runs
    a simplified DC->AC conversion with inverter clipping.
    """
    if tilt is None:
        tilt = abs(lat)  # Rule of thumb: tilt = latitude

    # Build a DataFrame pvlib expects
    times = pd.DatetimeIndex(timestamps_utc)
    weather = pd.DataFrame({"ghi": ghi_series.values}, index=times)
    weather["ghi"] = weather["ghi"].clip(lower=0)

    # Location and solar position
    loc = Location(lat, lon, tz="UTC")
    solpos = loc.get_solarposition(times)

    # Decompose GHI -> DNI, DHI using the Erbs model
    dni_dhi = pvlib.irradiance.erbs(weather["ghi"], solpos["zenith"], times)
    weather["dni"] = dni_dhi["dni"].clip(lower=0)
    weather["dhi"] = dni_dhi["dhi"].clip(lower=0)

    # Compute plane-of-array irradiance
    if tracking:
        # Single-axis tracker (horizontal N-S axis, typical in Texas)
        tracker = pvlib.tracking.singleaxis(
            apparent_zenith=solpos["apparent_zenith"],
            apparent_azimuth=solpos["azimuth"],
            max_angle=60,
            backtrack=True,
            gcr=0.35,  # Ground coverage ratio ~0.35 typical
        )
        surface_tilt = tracker["surface_tilt"].fillna(0)
        surface_azimuth = tracker["surface_azimuth"].fillna(180)
    else:
        surface_tilt = tilt
        surface_azimuth = 180  # South-facing

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=weather["dni"],
        ghi=weather["ghi"],
        dhi=weather["dhi"],
    )

    poa_global = poa["poa_global"].clip(lower=0).fillna(0)

    # Simple DC power model: P_dc = POA / 1000 * DC_capacity * derating
    # Use 0.85 derating to match the current model's "efficiency" parameter
    # (covers soiling, mismatch, wiring, temperature approximation)
    dc_capacity_mw = capacity_mw * dc_ac_ratio
    derating = 0.85
    dc_power = dc_capacity_mw * (poa_global / 1000.0) * derating

    # Inverter clipping at AC nameplate
    ac_power = dc_power.clip(upper=capacity_mw).clip(lower=0)

    return ac_power


def calculate_metrics(actual, modeled):
    """Calculate R, MBE, and RMSE between actual and modeled series."""
    combined = pd.DataFrame({"actual": actual, "modeled": modeled}).dropna()
    if combined.empty or len(combined) < 10:
        return {"R": np.nan, "R_Hourly": np.nan, "R_Daily": np.nan, "MBE": np.nan, "RMSE": np.nan}

    a = combined["actual"]
    m = combined["modeled"]

    # Exclude offline periods (actual ~0 but model > 5 MW)
    valid = ~((a < 0.5) & (m > 5.0))
    a = a[valid]
    m = m[valid]

    if len(a) < 10:
        return {"R": np.nan, "R_Hourly": np.nan, "R_Daily": np.nan, "MBE": np.nan, "RMSE": np.nan}

    r_15 = a.corr(m) if a.nunique() > 1 and m.nunique() > 1 else np.nan

    comb_f = combined[valid]
    hourly = comb_f.resample("h").mean().dropna()
    r_h = hourly["actual"].corr(hourly["modeled"]) if len(hourly) > 2 else np.nan

    daily = comb_f.resample("D").mean().dropna()
    r_d = daily["actual"].corr(daily["modeled"]) if len(daily) > 2 else np.nan

    return {
        "R": r_15,
        "R_Hourly": r_h,
        "R_Daily": r_d,
        "MBE": float((m - a).mean()),
        "RMSE": float(np.sqrt(((m - a) ** 2).mean())),
    }


def generate_pvlib_profile(year, lat, lon, capacity_mw, tracking, dc_ac_ratio):
    """Generate a full-year 15-min PVLib solar profile from Open-Meteo GHI."""
    df_om = fetch_tmy.get_openmeteo_data(year, lat, lon)
    if df_om.empty:
        return pd.Series(dtype=float)

    timestamps = pd.to_datetime(df_om["datetime"], utc=True)
    ghi = df_om["GHI_Wm2"].astype(float)
    ghi.index = timestamps

    ac_hourly = solar_pvlib_from_ghi(
        ghi_series=ghi,
        timestamps_utc=timestamps,
        lat=lat,
        lon=lon,
        capacity_mw=capacity_mw,
        tracking=tracking,
        dc_ac_ratio=dc_ac_ratio,
    )

    s_hourly = pd.Series(ac_hourly.values, index=timestamps, name="Gen_MW")
    s_hourly = s_hourly[~s_hourly.index.duplicated(keep="first")]

    # Resample to 15-min
    s_15 = s_hourly.resample("15min").interpolate(method="linear")

    # Reindex to full year aligned to Central
    target_cst = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:45", freq="15min", tz="US/Central")
    target_utc = target_cst.tz_convert("UTC")
    aligned = s_15.reindex(target_utc)
    return aligned.ffill().bfill().fillna(0)


def run_comparison():
    with open("ercot_assets.json") as f:
        assets = json.load(f)

    solar_projects = {k: v for k, v in assets.items() if v.get("tech") == "Solar"}
    print(f"Found {len(solar_projects)} solar projects\n")

    start_date = "2024-12-01"
    end_date = "2025-11-30"

    results = []

    for name, meta in solar_projects.items():
        r_id = meta["resource_name"]
        lat, lon = meta["lat"], meta["lon"]
        capacity = meta["capacity_mw"]
        tracking_type = meta.get("tracking_type", None)
        dc_ac = meta.get("dc_ac_ratio", 1.3)

        # Assume single-axis tracking unless explicitly fixed
        is_tracking = tracking_type != "fixed"

        print(f"=== {name} ({capacity:.0f} MW, tracking={is_tracking}, dc_ac={dc_ac:.2f}) ===")

        # Fetch actuals (cache only - don't fetch from ERCOT)
        df_actual = sced_fetcher.get_asset_period_data_cache_only(r_id, start_date, end_date)
        if df_actual.empty:
            print(f"  No cached actual data, skipping.\n")
            continue

        df_actual = df_actual.set_index("Time")

        # --- Current heuristic model ---
        p24_h = fetch_tmy.get_profile_for_year(2024, "Solar", capacity, lat=lat, lon=lon, tracking=is_tracking)
        p25_h = fetch_tmy.get_profile_for_year(2025, "Solar", capacity, lat=lat, lon=lon, tracking=is_tracking)
        prof_heuristic = pd.concat([p24_h, p25_h])
        prof_heuristic = prof_heuristic[~prof_heuristic.index.duplicated(keep="first")]

        # --- PVLib model ---
        p24_pv = generate_pvlib_profile(2024, lat, lon, capacity, tracking=is_tracking, dc_ac_ratio=dc_ac)
        p25_pv = generate_pvlib_profile(2025, lat, lon, capacity, tracking=is_tracking, dc_ac_ratio=dc_ac)
        prof_pvlib = pd.concat([p24_pv, p25_pv])
        prof_pvlib = prof_pvlib[~prof_pvlib.index.duplicated(keep="first")]

        # Align to actuals
        h_aligned = prof_heuristic.reindex(df_actual.index).fillna(0)
        pv_aligned = prof_pvlib.reindex(df_actual.index).fillna(0)

        # Apply curtailment (base point cap)
        if "Base_Point_MW" in df_actual.columns:
            bp = df_actual["Base_Point_MW"].fillna(np.inf).clip(lower=0)
            h_aligned = np.minimum(h_aligned, bp)
            pv_aligned = np.minimum(pv_aligned, bp)

        # Metrics
        m_h = calculate_metrics(df_actual["Actual_MW"], h_aligned)
        m_pv = calculate_metrics(df_actual["Actual_MW"], pv_aligned)

        print(f"  Heuristic: R={m_h['R']:.4f}  MBE={m_h['MBE']:.2f}  RMSE={m_h['RMSE']:.2f}")
        print(f"  PVLib:     R={m_pv['R']:.4f}  MBE={m_pv['MBE']:.2f}  RMSE={m_pv['RMSE']:.2f}")

        delta_r = (m_pv["R"] - m_h["R"]) if not (np.isnan(m_pv["R"]) or np.isnan(m_h["R"])) else np.nan
        if not np.isnan(delta_r):
            print(f"  Delta R:   {delta_r:+.4f} ({'PVLIB better' if delta_r > 0 else 'Heuristic better'})")
        print()

        results.append({
            "Project": name,
            "Capacity_MW": capacity,
            "Tracking": is_tracking,
            "Heuristic_R": m_h["R"],
            "Heuristic_R_Daily": m_h["R_Daily"],
            "Heuristic_MBE": m_h["MBE"],
            "Heuristic_RMSE": m_h["RMSE"],
            "PVLib_R": m_pv["R"],
            "PVLib_R_Daily": m_pv["R_Daily"],
            "PVLib_MBE": m_pv["MBE"],
            "PVLib_RMSE": m_pv["RMSE"],
            "Delta_R": delta_r,
        })

    # Summary
    if results:
        df = pd.DataFrame(results)

        # Filter to projects with valid R values
        valid = df.dropna(subset=["Heuristic_R", "PVLib_R"])

        print("\n" + "=" * 90)
        print("SUMMARY: PVLib vs Heuristic Solar Model")
        print("=" * 90)

        cols = ["Project", "Capacity_MW", "Tracking", "Heuristic_R", "PVLib_R", "Delta_R",
                "Heuristic_MBE", "PVLib_MBE", "Heuristic_RMSE", "PVLib_RMSE"]
        print(valid[cols].to_string(index=False, float_format="%.4f"))

        print(f"\nProjects compared: {len(valid)}")
        print(f"PVLib better R:    {(valid['Delta_R'] > 0).sum()}")
        print(f"Heuristic better:  {(valid['Delta_R'] < 0).sum()}")
        print(f"Mean Delta R:      {valid['Delta_R'].mean():+.4f}")
        print(f"Mean Heuristic R:  {valid['Heuristic_R'].mean():.4f}")
        print(f"Mean PVLib R:      {valid['PVLib_R'].mean():.4f}")
        print(f"Mean |Heur MBE|:   {valid['Heuristic_MBE'].abs().mean():.2f} MW")
        print(f"Mean |PVLib MBE|:  {valid['PVLib_MBE'].abs().mean():.2f} MW")
        print(f"Mean Heur RMSE:    {valid['Heuristic_RMSE'].mean():.2f} MW")
        print(f"Mean PVLib RMSE:   {valid['PVLib_RMSE'].mean():.2f} MW")

        # Save
        df.to_json("benchmark_pvlib_vs_heuristic.json", orient="records", indent=2)
        print(f"\nResults saved to benchmark_pvlib_vs_heuristic.json")


if __name__ == "__main__":
    run_comparison()
