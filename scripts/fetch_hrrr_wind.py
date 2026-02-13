#!/usr/bin/env python3
"""
Standalone HRRR 10m wind ingestion CLI.

This script is separate from the main forecast pipeline.
It writes cached parquet files under data_cache/hrrr/.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hrrr_ingestion import (  # noqa: E402
    HRRRIngestionConfig,
    HUB_LOCATIONS,
    fetch_hrrr_10m_wind_point,
    resolve_point_from_selector,
)


def _build_parser():
    parser = argparse.ArgumentParser(description="Fetch HRRR 10m wind for one or more points.")
    parser.add_argument("--start", required=True, help="Start timestamp/date (UTC if tz omitted).")
    parser.add_argument("--end", required=True, help="End timestamp/date (UTC if tz omitted).")
    parser.add_argument("--model", default="hrrr", help="Herbie model name (default: hrrr)")
    parser.add_argument("--product", default="sfc", help="Herbie product (default: sfc)")
    parser.add_argument("--forecast-hour", type=int, default=0, help="Forecast hour fxx (default: 0)")
    parser.add_argument("--cycle-hours", type=int, default=1, help="Cycle spacing in hours (default: 1)")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore existing cache and refetch.")

    target = parser.add_argument_group("Target")
    target.add_argument("--hub", help=f"Hub key ({', '.join(sorted(HUB_LOCATIONS.keys()))})")
    target.add_argument("--project", help="Project name from ercot_assets.json")
    target.add_argument("--lat", type=float, help="Latitude (requires --lon)")
    target.add_argument("--lon", type=float, help="Longitude (requires --lat)")
    target.add_argument("--all-hubs", action="store_true", help="Fetch all standard hub points.")
    return parser


def _validate_target_args(args):
    chosen = int(bool(args.hub)) + int(bool(args.project)) + int(bool(args.all_hubs))
    has_latlon = (args.lat is not None) or (args.lon is not None)
    if has_latlon and not (args.lat is not None and args.lon is not None):
        raise ValueError("Provide both --lat and --lon together.")
    if has_latlon:
        chosen += 1
    if chosen != 1:
        raise ValueError("Choose exactly one target selector: --hub, --project, --lat/--lon, or --all-hubs.")


def _run_for_target(name, lat, lon, args, config):
    df, cache_path = fetch_hrrr_10m_wind_point(
        start_time=args.start,
        end_time=args.end,
        lat=lat,
        lon=lon,
        config=config,
        force_refresh=args.force_refresh,
        repo_root=ROOT,
    )
    print(
        f"[{name}] rows={len(df)} "
        f"range={df['valid_time_utc'].min()} -> {df['valid_time_utc'].max()} "
        f"cache={cache_path}"
    )


def main():
    parser = _build_parser()
    args = parser.parse_args()
    _validate_target_args(args)

    config = HRRRIngestionConfig(
        model=args.model,
        product=args.product,
        forecast_hour=int(args.forecast_hour),
        cycle_hours=int(args.cycle_hours),
    )

    try:
        if args.all_hubs:
            for hub_key, (lat, lon) in HUB_LOCATIONS.items():
                _run_for_target(hub_key, lat, lon, args, config)
            return

        lat, lon, name = resolve_point_from_selector(
            hub=args.hub,
            project=args.project,
            lat=args.lat,
            lon=args.lon,
            assets_path=ROOT / "ercot_assets.json",
        )
        _run_for_target(name, lat, lon, args, config)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
