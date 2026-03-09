"""
QUANTAGRI LIVE MONITOR
----------------------
Keeps backtest metrics "live" by pulling the latest Sentinel-2 and
Sentinel-1 data on a daily or weekly schedule.

Instead of looping historical years, this script:
    1. Detects which commodity-region seasons are currently ACTIVE
       (today falls within their growing window)
    2. Pulls all new S2 scenes since the last successful run
    3. Recomputes the rolling season-to-date metrics
    4. Appends a timestamped row to a live_results.csv
    5. Optionally sends an alert if a yield surprise crosses a threshold

Run modes:
    python quantagri_live_monitor.py --mode daily
        → pulls last 2 days of scenes, updates all active seasons

    python quantagri_live_monitor.py --mode weekly
        → pulls last 8 days of scenes, updates all active seasons

    python quantagri_live_monitor.py --mode backfill --days 30
        → catch up on a gap (e.g. after Colab session expired)

    python quantagri_live_monitor.py --mode test
        → processes first active season only, 7-day window, dry run (no CSV write)
        → use to verify Planetary Computer connection and config without side effects

Schedule via cron (Linux/Mac) or Task Scheduler (Windows):
    # Daily at 06:00 UTC (after PC ingestion window closes)
    0 6 * * * cd /path/to/quantagri && python quantagri_live_monitor.py --mode daily

    # Or run in Colab on a timer — see CELL 10 in the notebook.

Output files (appended, never overwritten):
    quantagri_live_results.csv      ← one row per active-season update
    quantagri_live_alerts.csv       ← rows where surprise exceeds threshold
    quantagri_monitor.log           ← timestamped run log

Dependencies: same as the rest of the pipeline
    pip install planetary-computer pystac-client stackstac
               xarray rioxarray "dask[array]" numpy scipy pandas
"""

import os
import csv
import json
import time
import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import planetary_computer as pc
import pystac_client
import stackstac
from collections import defaultdict

from quantagri_commodity_config     import COMMODITY_SEASONS, GrowingSeason, get_season_date_range
from quantagri_spectral_velocity_pc import (
    get_spectral_audit, spatial_mean,
    PC_STAC_URL, TARGET_EPSG, bbox_from_geometry,
    COMPOSITE_FREQ, SCL_VALID
)
from quantagri_metrics_engine_pc    import (
    compute_peak_ndvi, compute_peak_lswi,
    compute_velocity_stats, compute_tercile_means,
    compute_yield_surprise
)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR  = "./quantagri_live"
STATE_FILE          = "monitor_state.json"   # tracks last-run timestamps
LIVE_RESULTS_CSV    = "quantagri_live_results.csv"
ALERTS_CSV          = "quantagri_live_alerts.csv"
LOG_FILE            = "quantagri_monitor.log"

# Alert thresholds — trigger an alert row when surprise exceeds these
SURPRISE_ALERT_BPS  = 1.5    # bpa — absolute value threshold
VELOCITY_ALERT      = 0.015  # dNDVI/day — strong green-up or stress signal

# Resolution for live monitoring — 100m is fast enough for daily checks
# Change to 20 or 10 for higher fidelity (slower, more RAM)
LIVE_RESOLUTION_M   = 100

# How many historical years to use for yield surprise calibration
# (loaded from official_yields.csv if available, else skipped)
CALIBRATION_YEARS   = 5


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, LOG_FILE)

    logger = logging.getLogger("quantagri_monitor")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        ch = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------------------------

def load_state(output_dir: str) -> dict:
    """Load last-run state from JSON. Returns empty dict on first run."""
    path = os.path.join(output_dir, STATE_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(state: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, STATE_FILE)
    with open(path, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def state_key(season: GrowingSeason) -> str:
    return f"{season.commodity}__{season.region_id}"


# ---------------------------------------------------------------------------
# SEASON ACTIVITY CHECK
# ---------------------------------------------------------------------------

def get_active_seasons(today: datetime) -> List[Tuple[GrowingSeason, str, str]]:
    """
    Return all seasons where today falls within the growing window.
    For year-crossing seasons (e.g. Mato Grosso Oct→Mar), checks both
    the current-year and prior-year start.

    Returns list of (season, season_start_str, season_end_str).
    """
    active = []
    year   = today.year

    for season in COMMODITY_SEASONS:
        for start_year in [year, year - 1]:
            s_date, e_date = get_season_date_range(season, start_year)
            s_ts = pd.Timestamp(s_date)
            e_ts = pd.Timestamp(e_date)

            if s_ts <= pd.Timestamp(today.date()) <= e_ts:
                active.append((season, s_date, e_date))
                break

    return active


# ---------------------------------------------------------------------------
# INCREMENTAL SCENE FETCH
# ---------------------------------------------------------------------------

def fetch_incremental_scenes(
    bbox:            list,
    since_date:      str,
    until_date:      str,
    max_cloud_pct:   int = 80
) -> list:
    """
    Fetch Sentinel-2 scenes between since_date and until_date.
    Deduplicates by date (least-cloudy tile per day).
    """
    catalog = pystac_client.Client.open(PC_STAC_URL, modifier=pc.sign_inplace)
    items   = list(catalog.search(
        collections = ['sentinel-2-l2a'],
        bbox        = bbox,
        datetime    = f"{since_date}/{until_date}",
        query       = {"eo:cloud_cover": {"lt": max_cloud_pct}},
    ).items())

    if not items:
        return []

    by_date = defaultdict(list)
    for item in items:
        by_date[item.datetime.strftime('%Y-%m-%d')].append(item)

    deduped = [
        min(scenes, key=lambda i: i.properties.get('eo:cloud_cover', 100))
        for scenes in by_date.values()
    ]
    deduped.sort(key=lambda i: i.datetime)
    return deduped


# ---------------------------------------------------------------------------
# SEASON-TO-DATE COMPOSITE
# ---------------------------------------------------------------------------

def compute_season_to_date(
    season:     GrowingSeason,
    s_date:     str,
    today_str:  str,
    resolution: int = LIVE_RESOLUTION_M,
    logger:     logging.Logger = None
) -> Optional[xr.Dataset]:
    """
    Fetch and composite all scenes from season start to today.
    Returns None if no valid data found.
    """
    try:
        composites = get_spectral_audit(
            geometry_tuple = season.geometry,
            start_date     = s_date,
            end_date       = today_str,
            resolution     = resolution,
            max_cloud_pct  = 80
        )
        if composites.time.size == 0:
            return None
        return composites
    except Exception as e:
        if logger:
            logger.warning(f"  Failed to fetch {season.region_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# LIVE METRICS ROW
# ---------------------------------------------------------------------------

def compute_live_metrics(
    season:          GrowingSeason,
    s_date:          str,
    today_str:       str,
    composites:      xr.Dataset,
    official_yields: Dict[int, float],
    historical_ndvi: Dict[int, float],
    logger:          logging.Logger
) -> dict:
    """
    Compute all season-to-date metrics. Returns a flat dict ready for CSV.
    """
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    current_year = pd.Timestamp(today_str).year

    base = {
        'run_timestamp':  now,
        'as_of_date':     today_str,
        'commodity':      season.commodity,
        'region_id':      season.region_id,
        'season_start':   s_date,
        'season_year':    current_year,
    }

    df = spatial_mean(composites)

    days_in = (pd.Timestamp(today_str) - pd.Timestamp(s_date)).days
    base['days_into_season']    = days_in
    base['season_pct_elapsed']  = round(min(days_in / 180 * 100, 100), 1)

    peak_ndvi_d = compute_peak_ndvi(df)
    peak_lswi_d = compute_peak_lswi(df)
    vel_d       = compute_velocity_stats(df)
    tercile_d   = compute_tercile_means(df)

    latest = df.dropna(subset=['NDVI']).tail(1)
    if not latest.empty:
        base['current_ndvi']          = round(float(latest['NDVI'].iloc[0]), 4)
        base['current_lswi']          = round(float(latest['LSWI'].iloc[0]), 4) if 'LSWI' in latest else None
        base['current_ndvi_velocity'] = round(float(latest['NDVI_velocity'].iloc[0]), 5) if 'NDVI_velocity' in latest and pd.notna(latest['NDVI_velocity'].iloc[0]) else None
        base['latest_composite_date'] = pd.Timestamp(latest['time'].iloc[0]).strftime('%Y-%m-%d')
    else:
        base['current_ndvi'] = base['current_lswi'] = base['current_ndvi_velocity'] = None
        base['latest_composite_date'] = None

    surp_d = {}
    if peak_ndvi_d.get('peak_ndvi') and len(historical_ndvi) >= 3:
        surp_d = compute_yield_surprise(
            peak_ndvi        = peak_ndvi_d['peak_ndvi'],
            year             = current_year,
            official_yields  = official_yields,
            historical_ndvi  = historical_ndvi
        )
    else:
        surp_d = {
            'rs_yield_proxy': None, 'official_forecast': None,
            'yield_surprise': None, 'surprise_pct': None,
            'calibration_r2': None, 'calibration_n': len(historical_ndvi)
        }

    base['n_composites_to_date'] = composites.time.size

    return {**base, **peak_ndvi_d, **peak_lswi_d, **vel_d, **tercile_d, **surp_d}


# ---------------------------------------------------------------------------
# ALERT DETECTION
# ---------------------------------------------------------------------------

def check_alerts(row: dict) -> List[str]:
    alerts = []

    surp = row.get('yield_surprise')
    if surp is not None and abs(surp) >= SURPRISE_ALERT_BPS:
        direction = 'BEARISH' if surp < 0 else 'BULLISH'
        alerts.append(
            f"YIELD_SURPRISE_{direction}: {row['commodity']}/{row['region_id']} "
            f"RS surprise = {surp:+.2f} bpa "
            f"({row['days_into_season']} days into season)"
        )

    vel = row.get('current_ndvi_velocity')
    if vel is not None and abs(vel) >= VELOCITY_ALERT:
        direction = 'STRESS' if vel < 0 else 'GREENUP'
        alerts.append(
            f"VELOCITY_{direction}: {row['commodity']}/{row['region_id']} "
            f"dNDVI/dt = {vel:+.4f}/day "
            f"as of {row.get('latest_composite_date', 'unknown')}"
        )

    return alerts


# ---------------------------------------------------------------------------
# CSV WRITERS
# ---------------------------------------------------------------------------

LIVE_COLUMNS = [
    'run_timestamp', 'as_of_date', 'commodity', 'region_id',
    'season_start', 'season_year', 'days_into_season', 'season_pct_elapsed',
    'n_composites_to_date', 'latest_composite_date',
    'current_ndvi', 'current_lswi', 'current_ndvi_velocity',
    'peak_ndvi', 'peak_ndvi_date', 'peak_lswi', 'peak_lswi_date',
    'velocity_mean', 'velocity_max', 'velocity_min', 'velocity_std',
    'tercile_mean_early', 'tercile_mean_mid', 'tercile_mean_late',
    'rs_yield_proxy', 'official_forecast', 'yield_surprise', 'surprise_pct',
    'calibration_r2', 'calibration_n',
]


def append_live_row(row: dict, output_dir: str) -> None:
    path         = os.path.join(output_dir, LIVE_RESULTS_CSV)
    write_header = not os.path.exists(path)
    extra_cols   = sorted(set(row.keys()) - set(LIVE_COLUMNS))
    fieldnames   = LIVE_COLUMNS + extra_cols

    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            w.writeheader()
        w.writerow(row)


def append_alert_rows(alerts: List[str], row: dict, output_dir: str) -> None:
    if not alerts:
        return
    path         = os.path.join(output_dir, ALERTS_CSV)
    write_header = not os.path.exists(path)

    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['timestamp', 'alert', 'commodity',
                                          'region_id', 'as_of_date',
                                          'yield_surprise', 'current_ndvi_velocity'],
                           extrasaction='ignore')
        if write_header:
            w.writeheader()
        for alert in alerts:
            w.writerow({
                'timestamp':             row['run_timestamp'],
                'alert':                 alert,
                'commodity':             row['commodity'],
                'region_id':             row['region_id'],
                'as_of_date':            row['as_of_date'],
                'yield_surprise':        row.get('yield_surprise'),
                'current_ndvi_velocity': row.get('current_ndvi_velocity'),
            })


# ---------------------------------------------------------------------------
# YIELD DATA LOADER
# ---------------------------------------------------------------------------

def load_official_yields(csv_path: str) -> Dict:
    data = {}
    if not os.path.exists(csv_path):
        return data
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            c   = r['commodity'].strip()
            reg = r['region_id'].strip()
            y   = int(r['year'].strip())
            val = float(r['official_yield'].strip())
            data.setdefault(c, {}).setdefault(reg, {})[y] = val
    return data


def get_historical_ndvi(
    live_csv:  str,
    commodity: str,
    region_id: str
) -> Dict[int, float]:
    """
    Build historical peak NDVI lookup from past live_results rows.
    Only uses rows where the season is complete (season_pct_elapsed >= 95).
    """
    if not os.path.exists(live_csv):
        return {}
    df = pd.read_csv(live_csv)
    df = df[
        (df['commodity']          == commodity) &
        (df['region_id']          == region_id) &
        (df['season_pct_elapsed'] >= 95) &
        (df['peak_ndvi'].notna())
    ]
    df = df.sort_values('run_timestamp').groupby('season_year').last().reset_index()
    return dict(zip(df['season_year'].astype(int), df['peak_ndvi'].astype(float)))


# ---------------------------------------------------------------------------
# MAIN MONITOR LOOP
# ---------------------------------------------------------------------------

def run_monitor(
    mode:          str,
    output_dir:    str,
    yield_csv:     str,
    backfill_days: int       = 8,
    commodities:   List[str] = None,
    resolution:    int       = LIVE_RESOLUTION_M,
    dry_run:       bool      = False,
) -> None:

    logger = setup_logging(output_dir)
    logger.info(f"{'='*60}")
    logger.info(f"QuantAgri Live Monitor — mode={mode}")
    if dry_run:
        logger.info("DRY RUN — no CSV writes, no state updates")
    logger.info(f"Output dir: {output_dir}")

    today     = datetime.now(timezone.utc)
    today_str = today.strftime('%Y-%m-%d')

    # Determine look-back window
    if mode == 'daily':
        lookback_days = 2
    elif mode == 'weekly':
        lookback_days = 8
    elif mode == 'backfill':
        lookback_days = backfill_days
    elif mode == 'test':
        lookback_days = 7
        dry_run       = True   # test always implies dry run
        logger.info("TEST MODE — processing first active season only, 7-day window")
    else:
        raise ValueError(f"Unknown mode: {mode}. Use daily, weekly, backfill, or test.")

    state      = load_state(output_dir)
    yield_data = load_official_yields(yield_csv)
    live_csv   = os.path.join(output_dir, LIVE_RESULTS_CSV)

    active = get_active_seasons(today)

    if commodities:
        active = [(s, sd, ed) for s, sd, ed in active if s.commodity in commodities]

    # TEST MODE — limit to first active season only
    if mode == 'test' and active:
        active = active[:1]
        logger.info(f"TEST MODE — limiting to: {active[0][0].commodity}/{active[0][0].region_id}")

    logger.info(f"Active seasons today ({today_str}): {len(active)}")
    for s, sd, ed in active:
        logger.info(f"  {s.commodity:<10} {s.region_id:<28} {sd} → {ed}")

    if not active:
        logger.info("No active seasons — nothing to do.")
        return

    total_alerts = []

    for season, s_date, e_date in active:
        sk = state_key(season)
        logger.info(f"\n--- {season.commodity.upper()} / {season.region_id} ---")

        last_run   = state.get(sk, {}).get('last_scene_date', None)
        since      = (today - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        bbox       = bbox_from_geometry(season.geometry)
        new_scenes = fetch_incremental_scenes(bbox, since, today_str)

        if not new_scenes and last_run:
            logger.info(f"  No new scenes since {since} — skipping")
            continue

        logger.info(f"  {len(new_scenes)} new scenes since {since}")
        logger.info(f"  Building season-to-date composites ({s_date} → {today_str})...")

        composites = compute_season_to_date(season, s_date, today_str,
                                            resolution, logger)

        if composites is None:
            logger.warning(f"  No valid composites — skipping")
            continue

        logger.info(f"  {composites.time.size} composites available")

        official_yields = yield_data.get(season.commodity, {}).get(season.region_id, {})
        historical_ndvi = get_historical_ndvi(live_csv, season.commodity, season.region_id)

        row = compute_live_metrics(
            season          = season,
            s_date          = s_date,
            today_str       = today_str,
            composites      = composites,
            official_yields = official_yields,
            historical_ndvi = historical_ndvi,
            logger          = logger
        )

        logger.info(
            f"  current_ndvi={row.get('current_ndvi')}  "
            f"velocity={row.get('current_ndvi_velocity')}  "
            f"surprise={row.get('yield_surprise')} bpa  "
            f"({row.get('days_into_season')} days in, "
            f"{row.get('season_pct_elapsed')}% elapsed)"
        )

        # Skip CSV writes and state updates in dry run / test mode
        if not dry_run:
            append_live_row(row, output_dir)

            alerts = check_alerts(row)
            if alerts:
                for a in alerts:
                    logger.warning(f"  ⚠️  ALERT: {a}")
                append_alert_rows(alerts, row, output_dir)
                total_alerts.extend(alerts)

            state.setdefault(sk, {})
            state[sk]['last_run']        = today_str
            state[sk]['last_scene_date'] = today_str
            state[sk]['last_ndvi']       = row.get('current_ndvi')
            state[sk]['last_surprise']   = row.get('yield_surprise')
            save_state(state, output_dir)
        else:
            logger.info("  DRY RUN — skipping CSV write and state update")
            alerts = check_alerts(row)
            if alerts:
                for a in alerts:
                    logger.info(f"  DRY RUN — would have fired alert: {a}")

        time.sleep(1.0)

    logger.info(f"\n{'='*60}")
    if dry_run:
        logger.info("TEST/DRY RUN complete — no files written.")
    else:
        logger.info(f"Run complete. {len(total_alerts)} alerts fired.")
        logger.info(f"Results → {os.path.join(output_dir, LIVE_RESULTS_CSV)}")
        if total_alerts:
            logger.info(f"Alerts  → {os.path.join(output_dir, ALERTS_CSV)}")


# ---------------------------------------------------------------------------
# DASHBOARD SUMMARY
# ---------------------------------------------------------------------------

def print_live_dashboard(output_dir: str) -> None:
    live_csv = os.path.join(output_dir, LIVE_RESULTS_CSV)
    if not os.path.exists(live_csv):
        print("No live results yet — run monitor first.")
        return

    df = pd.read_csv(live_csv)
    df = df.sort_values('run_timestamp').groupby(
        ['commodity', 'region_id']
    ).last().reset_index()

    print(f"\n{'='*100}")
    print(f"  QuantAgri Live Dashboard  —  as of {df['as_of_date'].max()}")
    print(f"{'='*100}")
    print(f"{'COMMODITY':<10} {'REGION':<28} {'NDVI':>6} {'VELOCITY':>12} "
          f"{'SURPRISE':>10} {'DAYS IN':>8} {'%ELAPSED':>9} {'COMPOSITES':>11}")
    print('-'*100)

    for _, row in df.sort_values(['commodity', 'region_id']).iterrows():
        surp     = row.get('yield_surprise')
        surp_str = f"{surp:+.2f} bpa" if pd.notna(surp) else "  n/a"
        vel      = row.get('current_ndvi_velocity')
        vel_str  = f"{vel:+.4f}/d"    if pd.notna(vel)  else "     n/a"
        ndvi     = row.get('current_ndvi')
        ndvi_str = f"{ndvi:.3f}"      if pd.notna(ndvi) else "  n/a"

        print(
            f"{row['commodity']:<10} {row['region_id']:<28} "
            f"{ndvi_str:>6} {vel_str:>12} {surp_str:>10} "
            f"{int(row['days_into_season']):>8} "
            f"{row['season_pct_elapsed']:>8.1f}% "
            f"{int(row['n_composites_to_date']):>11}"
        )

    print(f"{'='*100}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='QuantAgri Live Monitor')
    p.add_argument('--mode',
                   choices=['daily', 'weekly', 'backfill', 'test'],
                   default='daily',
                   help='Run mode: daily, weekly, backfill, or test (default: daily)')
    p.add_argument('--output_dir',
                   default=DEFAULT_OUTPUT_DIR,
                   help='Output directory for live results')
    p.add_argument('--yield_csv',
                   default='./official_yields.csv',
                   help='Path to official_yields.csv')
    p.add_argument('--commodities',
                   nargs='+',
                   default=None,
                   help='Filter to specific commodities e.g. --commodities corn soy')
    p.add_argument('--resolution',
                   type=int,
                   default=LIVE_RESOLUTION_M,
                   help='Spatial resolution in metres (default 100)')
    p.add_argument('--backfill_days',
                   type=int,
                   default=30,
                   help='Days to backfill when mode=backfill (default 30)')
    p.add_argument('--dashboard',
                   action='store_true',
                   help='Print live dashboard after run')
    p.add_argument('--dry_run',
                   action='store_true',
                   help='Skip CSV writes and state updates (implied by --mode test)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    run_monitor(
        mode          = args.mode,
        output_dir    = args.output_dir,
        yield_csv     = args.yield_csv,
        backfill_days = args.backfill_days,
        commodities   = args.commodities,
        resolution    = args.resolution,
        dry_run       = args.dry_run,
    )

    if args.dashboard:
        print_live_dashboard(args.output_dir)
