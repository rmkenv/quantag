"""
QUANTAGRI METRICS ENGINE (PLANETARY COMPUTER VERSION)
------------------------------------------------------
Operates on xarray Datasets + pandas DataFrames from spectral_velocity_pc.
All metric definitions unchanged from GEE version.
"""

import numpy as np
import pandas as pd
import xarray as xr
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('rasterio').setLevel(logging.ERROR)

import planetary_computer as pc
import pystac_client
import stackstac

from scipy.stats import pearsonr
from typing import Dict, List, Optional
from collections import defaultdict

from quantagri_spectral_velocity_pc import (
    PC_STAC_URL, TARGET_EPSG, bbox_from_geometry, spatial_mean
)

S1_COLLECTION = "sentinel-1-grd"
SAR_RES_M     = 40   # S1 IW native ~10m but 40m is safe for all ROI sizes


# ---------------------------------------------------------------------------
# 1. PEAK NDVI
# ---------------------------------------------------------------------------

def compute_peak_ndvi(df: pd.DataFrame) -> dict:
    if 'NDVI' not in df.columns or df['NDVI'].isna().all():
        return {'peak_ndvi': None, 'peak_ndvi_date': None}
    idx = df['NDVI'].idxmax()
    return {
        'peak_ndvi':      round(float(df.loc[idx, 'NDVI']), 4),
        'peak_ndvi_date': pd.Timestamp(df.loc[idx, 'time']).strftime('%Y-%m-%d')
    }


# ---------------------------------------------------------------------------
# 2. PEAK LSWI
# ---------------------------------------------------------------------------

def compute_peak_lswi(df: pd.DataFrame) -> dict:
    if 'LSWI' not in df.columns or df['LSWI'].isna().all():
        return {'peak_lswi': None, 'peak_lswi_date': None}
    idx = df['LSWI'].idxmax()
    return {
        'peak_lswi':      round(float(df.loc[idx, 'LSWI']), 4),
        'peak_lswi_date': pd.Timestamp(df.loc[idx, 'time']).strftime('%Y-%m-%d')
    }


# ---------------------------------------------------------------------------
# 3. VELOCITY STATS
# ---------------------------------------------------------------------------

def compute_velocity_stats(df: pd.DataFrame) -> dict:
    empty = {'velocity_mean': None, 'velocity_max': None,
             'velocity_min': None,  'velocity_std': None}
    if 'NDVI_velocity' not in df.columns:
        return empty
    v = df['NDVI_velocity'].dropna()
    if len(v) == 0:
        return empty
    return {
        'velocity_mean': round(float(v.mean()), 5),
        'velocity_max':  round(float(v.max()),  5),
        'velocity_min':  round(float(v.min()),  5),
        'velocity_std':  round(float(v.std()),  5),
    }


# ---------------------------------------------------------------------------
# 4. LEAD-TIME R²
# ---------------------------------------------------------------------------

def compute_tercile_means(df: pd.DataFrame) -> dict:
    """
    Split one season's NDVI time series into three equal terciles,
    return mean NDVI per tercile. Called once per year.
    """
    empty = {'tercile_mean_early': None, 'tercile_mean_mid': None,
             'tercile_mean_late': None}
    if df.empty or 'NDVI' not in df.columns:
        return empty

    times = pd.to_datetime(df['time'])
    t0, t3 = times.min(), times.max()
    if t0 == t3:
        return empty

    span = (t3 - t0).total_seconds()
    t1   = t0 + pd.Timedelta(seconds=span / 3)
    t2   = t0 + pd.Timedelta(seconds=span * 2 / 3)

    df2 = df.copy()
    df2['time'] = times
    early = df2[df2['time'] <= t1]['NDVI'].mean()
    mid   = df2[(df2['time'] > t1) & (df2['time'] <= t2)]['NDVI'].mean()
    late  = df2[df2['time'] > t2]['NDVI'].mean()

    return {
        'tercile_mean_early': round(float(early), 4) if pd.notna(early) else None,
        'tercile_mean_mid':   round(float(mid),   4) if pd.notna(mid)   else None,
        'tercile_mean_late':  round(float(late),  4) if pd.notna(late)  else None,
    }


def compute_r2_across_years(
    per_year_terciles: Dict[int, dict],
    official_yields:   Dict[int, float]
) -> dict:
    """Cross-year R² per tercile. Requires ≥3 years with both NDVI and yield data."""
    results = {}
    for label in ('early', 'mid', 'late'):
        key   = f'tercile_mean_{label}'
        years = sorted(
            y for y in per_year_terciles
            if y in official_yields and per_year_terciles[y].get(key) is not None
        )
        if len(years) < 3:
            results.update({f'r2_{label}': None, f'p_{label}': None, f'n_{label}': len(years)})
            continue
        x = np.array([per_year_terciles[y][key]  for y in years])
        y = np.array([official_yields[y]          for y in years])
        r, p = pearsonr(x, y)
        results.update({
            f'r2_{label}': round(float(r**2), 4),
            f'p_{label}':  round(float(p),    4),
            f'n_{label}':  len(years)
        })
    return results


# ---------------------------------------------------------------------------
# 5. YIELD SURPRISE
# ---------------------------------------------------------------------------

def compute_yield_surprise(
    peak_ndvi:       float,
    year:            int,
    official_yields: Dict[int, float],
    historical_ndvi: Dict[int, float]
) -> dict:
    """OLS calibration of NDVI → yield, then compute surprise vs official forecast."""
    cal_years = [y for y in historical_ndvi if y != year and y in official_yields]
    if len(cal_years) < 3:
        return {
            'rs_yield_proxy': None, 'official_forecast': official_yields.get(year),
            'yield_surprise': None, 'surprise_pct': None,
            'calibration_r2': None, 'calibration_n': len(cal_years)
        }
    x  = np.array([historical_ndvi[y] for y in cal_years])
    yy = np.array([official_yields[y] for y in cal_years])
    b, a   = np.polyfit(x, yy, 1)
    r, _   = pearsonr(x, yy)
    proxy  = float(a + b * peak_ndvi)
    fc     = official_yields.get(year)
    surp   = round(proxy - fc, 4)       if fc else None
    surp_p = round(surp / fc * 100, 2)  if fc else None
    return {
        'rs_yield_proxy':    round(proxy, 4),
        'official_forecast': fc,
        'yield_surprise':    surp,
        'surprise_pct':      surp_p,
        'calibration_r2':    round(float(r**2), 4),
        'calibration_n':     len(cal_years)
    }


# ---------------------------------------------------------------------------
# 6. CLOUD COVER  (STAC metadata — no raster download needed)
# ---------------------------------------------------------------------------

def compute_avg_cloud_cover(
    geometry_tuple: tuple, start_date: str, end_date: str
) -> dict:
    """Scene-level cloud cover from STAC item properties. Fast — no pixel loading."""
    bbox    = bbox_from_geometry(geometry_tuple)
    catalog = pystac_client.Client.open(PC_STAC_URL, modifier=pc.sign_inplace)
    items   = list(catalog.search(
        collections=['sentinel-2-l2a'], bbox=bbox,
        datetime=f"{start_date}/{end_date}"
    ).items())

    if not items:
        return {'cloud_cover_mean': None, 'cloud_cover_max': None, 'n_scenes': 0}

    cc = np.array([
        item.properties.get('eo:cloud_cover', np.nan)
        for item in items
        if item.properties.get('eo:cloud_cover') is not None
    ])
    if len(cc) == 0:
        return {'cloud_cover_mean': None, 'cloud_cover_max': None, 'n_scenes': len(items)}

    return {
        'cloud_cover_mean': round(float(cc.mean()) / 100, 3),
        'cloud_cover_max':  round(float(cc.max())  / 100, 3),
        'n_scenes':         len(items)
    }


# ---------------------------------------------------------------------------
# 7. SAR (Sentinel-1 GRD via PC)
# ---------------------------------------------------------------------------

def compute_sar_metrics(
    geometry_tuple: tuple, start_date: str, end_date: str, commodity: str
) -> dict:
    """
    Sentinel-1 IW descending VV/VH backscatter.
    Returns dB-scale statistics. VH/VV ratio = canopy volume proxy.
    """
    empty = {
        'sar_vv_mean': None, 'sar_vv_std': None,
        'sar_vh_mean': None, 'sar_vh_max': None,
        'sar_vh_vv_ratio': None, 'sar_n_acquisitions': 0
    }
    bbox    = bbox_from_geometry(geometry_tuple)
    catalog = pystac_client.Client.open(PC_STAC_URL, modifier=pc.sign_inplace)

    # Note: PC S1 query uses 'sat:orbit_state' not 'orbitProperties_pass'
    items = list(catalog.search(
        collections = [S1_COLLECTION],
        bbox        = bbox,
        datetime    = f"{start_date}/{end_date}",
        query       = {
            "s1:instrument_mode": {"eq": "IW"},
            "sat:orbit_state":    {"eq": "descending"}
        }
    ).items())

    if not items:
        print(f"    [SAR] No S1 scenes — {commodity} {start_date}→{end_date}")
        return empty

    # Deduplicate by date (same logic as S2)
    by_date = defaultdict(list)
    for item in items:
        by_date[item.datetime.strftime('%Y-%m-%d')].append(item)
    items = [scenes[0] for scenes in by_date.values()]
    items.sort(key=lambda i: i.datetime)

    print(f"    [SAR] {len(items)} S1 acquisitions")
    empty['sar_n_acquisitions'] = len(items)

    # Re-sign SAR items too — same SAS token expiry risk
    items = [pc.sign(item) for item in items]

    try:
        s1_stack = stackstac.stack(
            items,
            assets        = ['VV', 'VH'],
            resolution    = SAR_RES_M,
            epsg          = TARGET_EPSG,
            bounds_latlon = bbox,
            dtype         = float,
            rescale       = False,
            chunksize     = 1024,
        )
        s1_ds = s1_stack.to_dataset(dim='band')

        # Convert linear → dB
        s1_ds['VV_db'] = 10 * np.log10(s1_ds['VV'].clip(min=1e-10))
        s1_ds['VH_db'] = 10 * np.log10(s1_ds['VH'].clip(min=1e-10))
        s1_ds['ratio'] = s1_ds['VH_db'] - s1_ds['VV_db']

        vv = s1_ds['VV_db'].mean(dim=['x','y']).compute().values
        vh = s1_ds['VH_db'].mean(dim=['x','y']).compute().values
        rt = s1_ds['ratio'].mean(dim=['x','y']).compute().values

        vv = vv[~np.isnan(vv)]; vh = vh[~np.isnan(vh)]; rt = rt[~np.isnan(rt)]

        return {
            'sar_vv_mean':        round(float(vv.mean()), 3) if len(vv) else None,
            'sar_vv_std':         round(float(vv.std()),  3) if len(vv) else None,
            'sar_vh_mean':        round(float(vh.mean()), 3) if len(vh) else None,
            'sar_vh_max':         round(float(vh.max()),  3) if len(vh) else None,
            'sar_vh_vv_ratio':    round(float(rt.mean()), 3) if len(rt) else None,
            'sar_n_acquisitions': len(items)
        }
    except Exception as e:
        print(f"    [SAR WARNING] {e}")
        return {**empty, 'sar_n_acquisitions': len(items)}


# ---------------------------------------------------------------------------
# MASTER FUNCTION
# ---------------------------------------------------------------------------

def compute_all_metrics(
    composites:         xr.Dataset,
    geometry_tuple:     tuple,
    start_date:         str,
    end_date:           str,
    commodity:          str,
    region_id:          str,
    year:               int,
    official_yields:    Dict[int, float],
    historical_ndvi:    Dict[int, float],
    per_year_terciles:  dict
) -> dict:
    """Compute full metric suite for one commodity-region-year."""
    base = {'commodity': commodity, 'region_id': region_id, 'year': year}

    print(f"    Computing spatial means...")
    df = spatial_mean(composites)

    peak_ndvi_d = compute_peak_ndvi(df)
    peak_lswi_d = compute_peak_lswi(df)
    vel_d       = compute_velocity_stats(df)

    # Tercile means for this year — stored for cross-year R²
    tercile_d = compute_tercile_means(df)
    per_year_terciles[year] = tercile_d

    r2_d = compute_r2_across_years(per_year_terciles, official_yields)

    surp_d = {}
    if peak_ndvi_d.get('peak_ndvi') is not None:
        surp_d = compute_yield_surprise(
            peak_ndvi_d['peak_ndvi'], year, official_yields, historical_ndvi
        )

    print(f"    Computing cloud cover...")
    cloud_d = compute_avg_cloud_cover(geometry_tuple, start_date, end_date)

    print(f"    Computing SAR...")
    sar_d = compute_sar_metrics(geometry_tuple, start_date, end_date, commodity)

    return {**base, **peak_ndvi_d, **peak_lswi_d, **vel_d,
            **r2_d, **surp_d, **cloud_d, **sar_d}
