"""
QUANTAGRI HISTORICAL MONTHLY AGGREGATOR
-----------------------------------------
Pulls 10 years of Sentinel-2 data (2016-2025) for all commodity-region
growing seasons and aggregates metrics by MONTH.

Output: quantagri_historical/quantagri_monthly_COMMODITY_REGION.csv
        quantagri_historical/quantagri_monthly_ALL.csv  (combined)

Monthly columns per region:
    year, month, month_label,
    ndvi_mean, ndvi_max, ndvi_min, ndvi_std,
    lswi_mean, lswi_max,
    velocity_mean, velocity_max, velocity_min,
    n_composites, cloud_cover_mean

This is designed to run as a one-time GitHub Actions job (or re-run
whenever you want to refresh the historical baseline).

Usage:
    python quantagri_historical_monthly.py --start_year 2016 --end_year 2025
    python quantagri_historical_monthly.py --commodities soy wheat --start_year 2016
"""

import os
import csv
import time
import argparse
import warnings
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')
logging.getLogger('rasterio').setLevel(logging.ERROR)

from quantagri_commodity_config import (
    COMMODITY_SEASONS, GrowingSeason, get_season_date_range, list_seasons
)
from quantagri_spectral_velocity_pc import (
    get_spectral_audit, spatial_mean
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DEFAULT_START_YEAR  = 2016   # Sentinel-2 launched mid-2015; 2016 = first full year
DEFAULT_END_YEAR    = 2025
DEFAULT_OUTPUT_DIR  = "./quantagri_historical"
DEFAULT_RESOLUTION  = 200    # 200m for historical — balances speed vs fidelity
                              # 10 years × 12 regions = ~120 season pulls

# Per-commodity resolution overrides — large ROIs need coarser resolution
# to fit in the 7GB GitHub Actions runner RAM.
# These mirror the daily workflow resolutions.
COMMODITY_RESOLUTION = {
    'soy':    500,   # Mato Grosso — huge ROI, OOMs at 200m
    'sugar':  300,   # Sao Paulo + Uttar Pradesh — two large tropical regions
    'corn':   200,
    'wheat':  200,
    'cotton': 200,
}


# ---------------------------------------------------------------------------
# MONTHLY AGGREGATION
# Takes a season-level xr.Dataset and breaks it into calendar months,
# computing mean/max/min/std of NDVI, LSWI, velocity per month.
# ---------------------------------------------------------------------------

def aggregate_by_month(
    composites: xr.Dataset,
    season:     GrowingSeason,
    year:       int
) -> List[dict]:
    """
    Break a season's composites into calendar months and aggregate.
    Returns a list of dicts — one per month that has data.
    """
    df = spatial_mean(composites)
    if df.empty:
        return []

    df['time'] = pd.to_datetime(df['time'])
    df['year']  = df['time'].dt.year
    df['month'] = df['time'].dt.month

    rows = []
    for (yr, mo), group in df.groupby(['year', 'month']):
        ndvi = group['NDVI'].dropna()
        lswi = group['LSWI'].dropna() if 'LSWI' in group else pd.Series(dtype=float)
        vel  = group['NDVI_velocity'].dropna() if 'NDVI_velocity' in group else pd.Series(dtype=float)

        rows.append({
            'commodity':      season.commodity,
            'region_id':      season.region_id,
            'season_year':    year,          # year the season STARTED
            'year':           int(yr),       # calendar year of this month
            'month':          int(mo),
            'month_label':    pd.Timestamp(year=int(yr), month=int(mo), day=1).strftime('%b %Y'),

            # NDVI
            'ndvi_mean':      round(float(ndvi.mean()), 4) if len(ndvi) else None,
            'ndvi_max':       round(float(ndvi.max()),  4) if len(ndvi) else None,
            'ndvi_min':       round(float(ndvi.min()),  4) if len(ndvi) else None,
            'ndvi_std':       round(float(ndvi.std()),  4) if len(ndvi) > 1 else None,

            # LSWI
            'lswi_mean':      round(float(lswi.mean()), 4) if len(lswi) else None,
            'lswi_max':       round(float(lswi.max()),  4) if len(lswi) else None,

            # Velocity
            'velocity_mean':  round(float(vel.mean()), 5) if len(vel) else None,
            'velocity_max':   round(float(vel.max()),  5) if len(vel) else None,
            'velocity_min':   round(float(vel.min()),  5) if len(vel) else None,

            # Coverage
            'n_composites':   len(group),
        })

    return rows


# ---------------------------------------------------------------------------
# CSV HELPERS
# ---------------------------------------------------------------------------

MONTHLY_COLUMNS = [
    'commodity', 'region_id', 'season_year', 'year', 'month', 'month_label',
    'ndvi_mean', 'ndvi_max', 'ndvi_min', 'ndvi_std',
    'lswi_mean', 'lswi_max',
    'velocity_mean', 'velocity_max', 'velocity_min',
    'n_composites',
]


def append_rows(rows: List[dict], path: str) -> None:
    if not rows:
        return
    write_header = not os.path.exists(path)
    extra = sorted(set(k for r in rows for k in r) - set(MONTHLY_COLUMNS))
    fieldnames = MONTHLY_COLUMNS + extra
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            w.writeheader()
        w.writerows(rows)


def write_combined(output_dir: str) -> None:
    """Merge all per-commodity CSVs into one combined file."""
    all_rows = []
    for fname in os.listdir(output_dir):
        if fname.startswith('quantagri_monthly_') and fname.endswith('.csv') \
                and fname != 'quantagri_monthly_ALL.csv':
            path = os.path.join(output_dir, fname)
            try:
                df = pd.read_csv(path)
                all_rows.append(df)
            except Exception:
                pass
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined = combined.sort_values(['commodity', 'region_id', 'season_year', 'month'])
        combined.to_csv(os.path.join(output_dir, 'quantagri_monthly_ALL.csv'), index=False)
        print(f"[COMBINED] {len(combined)} rows → quantagri_monthly_ALL.csv")


# ---------------------------------------------------------------------------
# STATE — track which season-years are already done so re-runs are safe
# ---------------------------------------------------------------------------

def load_completed(output_dir: str) -> set:
    """Return set of 'commodity__region__year' already in the output files."""
    done = set()
    for fname in os.listdir(output_dir):
        if not fname.startswith('quantagri_monthly_') or not fname.endswith('.csv'):
            continue
        if fname == 'quantagri_monthly_ALL.csv':
            continue
        path = os.path.join(output_dir, fname)
        try:
            df = pd.read_csv(path, usecols=['commodity', 'region_id', 'season_year'])
            for _, row in df.drop_duplicates().iterrows():
                done.add(f"{row['commodity']}__{row['region_id']}__{int(row['season_year'])}")
        except Exception:
            pass
    return done


# ---------------------------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------------------------

def run_historical(
    start_year:  int,
    end_year:    int,
    commodities: Optional[List[str]],
    output_dir:  str,
    resolution:  int,
    resume:      bool = True
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    seasons = [s for s in COMMODITY_SEASONS
               if not commodities or s.commodity in commodities]

    completed = load_completed(output_dir) if resume else set()
    if completed:
        print(f"[RESUME] {len(completed)} season-years already done — skipping")

    total   = len(seasons) * (end_year - start_year + 1)
    counter = 0

    for season in seasons:
        rk       = f"{season.commodity}__{season.region_id}"
        out_path = os.path.join(output_dir,
                                f"quantagri_monthly_{season.commodity}_{season.region_id}.csv")

        print(f"\n{'='*60}")
        print(f"  {season.commodity.upper()} — {season.label}")
        print(f"{'='*60}")

        # Use per-commodity resolution override if available
        effective_resolution = COMMODITY_RESOLUTION.get(season.commodity, resolution)
        if effective_resolution != resolution:
            print(f"  [RESOLUTION] Using {effective_resolution}m for {season.commodity} (override)")

        for year in range(start_year, end_year + 1):
            counter += 1
            key = f"{rk}__{year}"

            if key in completed:
                print(f"  [{counter}/{total}] {key} — already done, skipping")
                continue

            print(f"\n  [{counter}/{total}] {rk} — {year}")

            try:
                s_date, e_date = get_season_date_range(season, year)
                print(f"    {s_date} → {e_date}")

                composites = get_spectral_audit(
                    geometry_tuple = season.geometry,
                    start_date     = s_date,
                    end_date       = e_date,
                    resolution     = effective_resolution,
                    max_cloud_pct  = 80
                )

                if composites.time.size == 0:
                    print(f"    [SKIP] No composites")
                    continue

                print(f"    {composites.time.size} composites → aggregating by month...")

                monthly_rows = aggregate_by_month(composites, season, year)

                if not monthly_rows:
                    print(f"    [SKIP] No monthly rows produced")
                    continue

                append_rows(monthly_rows, out_path)
                completed.add(key)

                months_found = sorted(set(r['month_label'] for r in monthly_rows))
                print(f"    ✓ {len(monthly_rows)} month-rows: {', '.join(months_found)}")

            except MemoryError:
                # Retry at double the resolution (half the pixels)
                retry_res = effective_resolution * 2
                print(f"    [MEM ERROR] Retrying at {retry_res}m...")
                try:
                    composites = get_spectral_audit(
                        geometry_tuple = season.geometry,
                        start_date     = s_date,
                        end_date       = e_date,
                        resolution     = retry_res,
                        max_cloud_pct  = 80
                    )
                    if composites.time.size > 0:
                        monthly_rows = aggregate_by_month(composites, season, year)
                        if monthly_rows:
                            append_rows(monthly_rows, out_path)
                            completed.add(key)
                            print(f"    ✓ Retry succeeded at {retry_res}m: {len(monthly_rows)} month-rows")
                except Exception as e2:
                    print(f"    [RETRY FAILED] {e2}")
            except Exception as e:
                print(f"    [ERROR] {type(e).__name__}: {e}")

            time.sleep(0.5)

    # Write combined file at end
    write_combined(output_dir)
    print(f"\n[DONE] Output: {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='QuantAgri 10-Year Monthly Aggregator')
    p.add_argument('--start_year',  type=int, default=DEFAULT_START_YEAR)
    p.add_argument('--end_year',    type=int, default=DEFAULT_END_YEAR)
    p.add_argument('--commodities', nargs='+', default=None,
                   help='e.g. --commodities corn soy wheat')
    p.add_argument('--output_dir',  default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--resolution',  type=int, default=DEFAULT_RESOLUTION)
    p.add_argument('--no_resume',   action='store_true',
                   help='Re-run everything even if already completed')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    list_seasons()
    run_historical(
        start_year  = args.start_year,
        end_year    = args.end_year,
        commodities = args.commodities,
        output_dir  = args.output_dir,
        resolution  = args.resolution,
        resume      = not args.no_resume
    )
