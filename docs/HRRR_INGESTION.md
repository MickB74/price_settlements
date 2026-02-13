# HRRR Ingestion (Separate Path)

This repo now includes a standalone HRRR ingestion path that is intentionally separate from the default forecast engine.

Files:
- `/Users/michaelbarry/Documents/GitHub/price_settlements/utils/hrrr_ingestion.py`
- `/Users/michaelbarry/Documents/GitHub/price_settlements/scripts/fetch_hrrr_wind.py`

## Purpose

Fetch and cache HRRR 10m wind for selected points (hubs/projects/lat-lon) for offline comparison and future integration testing.

Output cache location:
- `/Users/michaelbarry/Documents/GitHub/price_settlements/data_cache/hrrr/`

## Optional Dependencies

HRRR ingestion uses `Herbie` and GRIB decoding backends:

```bash
python -m pip install herbie cfgrib eccodes
```

If not installed, the script raises a clear runtime error and leaves the default model path unchanged.

## Usage

Single hub:

```bash
python scripts/fetch_hrrr_wind.py \
  --start 2025-01-01 \
  --end 2025-01-07 \
  --hub HB_WEST
```

Single project from `ercot_assets.json`:

```bash
python scripts/fetch_hrrr_wind.py \
  --start 2025-01-01 \
  --end 2025-01-07 \
  --project "Azure Sky Wind"
```

Explicit coordinate:

```bash
python scripts/fetch_hrrr_wind.py \
  --start 2025-01-01 \
  --end 2025-01-07 \
  --lat 32.4518 \
  --lon -100.5371
```

All ERCOT hubs:

```bash
python scripts/fetch_hrrr_wind.py \
  --start 2025-01-01 \
  --end 2025-01-03 \
  --all-hubs
```

Advanced options:
- `--forecast-hour` (default `0`)
- `--cycle-hours` (default `1`)
- `--product` (default `sfc`)
- `--model` (default `hrrr`)
- `--force-refresh`

## Notes

- Streamlit now exposes wind dataset selectors with:
  - `Open-Meteo / PVGIS (Default)`
  - `NOAA HRRR (Cached)`
- The HRRR option reads from local cache only and falls back to default weather if cache is missing.
- Cache files are parquet plus metadata JSON sidecars.
