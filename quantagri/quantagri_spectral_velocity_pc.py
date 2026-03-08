"""
QUANTAGRI SPECTRAL VELOCITY ENGINE (PLANETARY COMPUTER IMPLEMENTATION)
-----------------------------------------------------------------------
Drop-in replacement for the GEE version. Uses Microsoft Planetary Computer
(free, no waitlist) via pystac-client + stackstac + xarray.

Dependencies:
    pip install planetary-computer pystac-client stackstac \
                xarray rioxarray "dask[array]" numpy scipy pandas

PC STAC endpoint : https://planetarycomputer.microsoft.com/api/stac/v1
Collection       : sentinel-2-l2a  (equivalent to GEE S2_SR_HARMONIZED)

References:
    Gao, B.-C. (1996). NDWI. Remote Sensing of Environment, 58(3), 257-266.
    ESA Sentinel-2 L2A Algorithm Theoretical Basis Document.
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
from collections import defaultdict


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

PC_STAC_URL    = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION  = "sentinel-2-l2a"

# EPSG:3857 (Web Mercator) — resolution is in METRES.
# DO NOT use EPSG:4326 here: that CRS uses degrees, so resolution=10
# means 10 degrees/pixel and the entire state of Iowa becomes 1 pixel.
TARGET_EPSG    = 3857
TARGET_RES_M   = 100           # 100m default — safe for Colab free tier RAM.
                                # Override with resolution=10 for full precision
                                # only on small ROIs or Colab Pro (25GB RAM).
COMPOSITE_FREQ = "16D"
WARMUP_DAYS    = 16

# SCL classes to KEEP: 4=Vegetation 5=BareSoil 6=Water 7=Unclassified
# Excluded: 0=NoData 1=Saturated 2=Dark 3=Shadow
#           8=MedCloud 9=HighCloud 10=Cirrus 11=Snow
SCL_VALID      = [4, 5, 6, 7]


# ---------------------------------------------------------------------------
# BBOX HELPER
# ---------------------------------------------------------------------------

def bbox_from_geometry(geometry_tuple: tuple) -> list:
    """
    (min_lat, min_lon, max_lat, max_lon)  →  [min_lon, min_lat, max_lon, max_lat]
    STAC uses lon/lat order (x before y).
    """
    min_lat, min_lon, max_lat, max_lon = geometry_tuple
    return [min_lon, min_lat, max_lon, max_lat]


# ---------------------------------------------------------------------------
# CLOUD MASKING
# ---------------------------------------------------------------------------

def mask_clouds_scl(ds: xr.Dataset) -> xr.Dataset:
    """
    Mask cloud/shadow pixels using the SCL band.
    Sets invalid pixels to NaN across B04, B08, B11.
    SCL is more accurate than QA60: handles shadows, cirrus, snow separately.
    """
    if 'SCL' not in ds:
        # Fallback: no masking if SCL missing (older granules)
        return ds

    scl   = ds['SCL']
    valid = xr.zeros_like(scl, dtype=bool)
    for cls in SCL_VALID:
        valid = valid | (scl == cls)

    for band in [v for v in ds.data_vars if v != 'SCL']:
        ds[band] = ds[band].where(valid)

    return ds


# ---------------------------------------------------------------------------
# SPECTRAL INDICES
# ---------------------------------------------------------------------------

def calculate_indices(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute NDVI and LSWI.
    PC stores raw DN (0-10000) — we divide by 10000 since rescale=False.

    NDVI = (B08 - B04) / (B08 + B04)
    LSWI = (B08 - B11) / (B08 + B11)  [Gao 1996 — not McFeeters NDWI]
    """
    # Guard: if bands are missing return ds unchanged
    for band in ['B04', 'B08', 'B11']:
        if band not in ds:
            raise KeyError(f"Band {band} missing from dataset. "
                           f"Available: {list(ds.data_vars)}")

    b04 = ds['B04'].astype(float) / 10000.0
    b08 = ds['B08'].astype(float) / 10000.0
    b11 = ds['B11'].astype(float) / 10000.0

    eps = 1e-10
    ds['NDVI'] = ((b08 - b04) / (b08 + b04 + eps)).clip(-1.0, 1.0)
    ds['LSWI'] = ((b08 - b11) / (b08 + b11 + eps)).clip(-1.0, 1.0)

    return ds


# ---------------------------------------------------------------------------
# COMPOSITING
# ---------------------------------------------------------------------------

def build_composites(ds: xr.Dataset, freq: str = COMPOSITE_FREQ) -> xr.Dataset:
    """
    16-day median composites over NDVI and LSWI.
    xarray resample().median() handles uneven scene counts per window gracefully.
    skipna=True means windows with partial cloud cover still produce a composite.
    """
    return ds[['NDVI', 'LSWI']].resample(time=freq).median(
        dim='time', skipna=True, keep_attrs=True
    )


# ---------------------------------------------------------------------------
# VELOCITY
# ---------------------------------------------------------------------------

def calculate_velocity(composites: xr.Dataset) -> xr.Dataset:
    """
    Spectral velocity: v(t) = NDVI(t) - NDVI(t-1).
    First timestep is NaN by design — no predecessor exists.
    xarray .diff() is the correct, safe way to do this (no off-by-one risk).
    """
    composites['NDVI_velocity'] = composites['NDVI'].diff(dim='time', label='upper')
    return composites


# ---------------------------------------------------------------------------
# PRIMARY ENTRY POINT
# ---------------------------------------------------------------------------

def get_spectral_audit(
    geometry_tuple: tuple,
    start_date:     str,
    end_date:       str,
    resolution:     int = TARGET_RES_M,
    max_cloud_pct:  int = 80
) -> xr.Dataset:
    """
    Full spectral audit pipeline via Planetary Computer.

    Args:
        geometry_tuple : (min_lat, min_lon, max_lat, max_lon)
        start_date     : 'YYYY-MM-DD'
        end_date       : 'YYYY-MM-DD'
        resolution     : Spatial resolution in METRES (default 100m).
                         Use 10 for full precision on small ROIs only.
        max_cloud_pct  : Scene-level cloud filter (0-100, default 80).

    Returns:
        xr.Dataset with time, y, x dims and NDVI, LSWI, NDVI_velocity vars.
    """
    # Warm-up buffer for velocity lag
    warmup_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=WARMUP_DAYS)
    ).strftime('%Y-%m-%d')
    date_range = f"{warmup_start}/{end_date}"
    bbox = bbox_from_geometry(geometry_tuple)

    # ── 1. STAC query ──────────────────────────────────────────────────
    catalog = pystac_client.Client.open(PC_STAC_URL, modifier=pc.sign_inplace)

    search = catalog.search(
        collections = [S2_COLLECTION],
        bbox        = bbox,
        datetime    = date_range,
        query       = {"eo:cloud_cover": {"lt": max_cloud_pct}},
    )
    items = list(search.items())

    if not items:
        raise ValueError(
            f"No S2 scenes found. bbox={bbox}, dates={date_range}"
        )

    # ── 2. Deduplicate tiles ────────────────────────────────────────────
    # Iowa intersects ~6 Sentinel-2 tiles. Without deduplication the same
    # acquisition date appears 6x. Keep least-cloudy scene per date.
    by_date = defaultdict(list)
    for item in items:
        by_date[item.datetime.strftime('%Y-%m-%d')].append(item)

    items = [
        min(scenes, key=lambda i: i.properties.get('eo:cloud_cover', 100))
        for scenes in by_date.values()
    ]
    items.sort(key=lambda i: i.datetime)
    print(f"  {len(items)} scenes after deduplication ({warmup_start} → {end_date})")

    # ── 3. Re-sign items before stacking ───────────────────────────────
    # SAS tokens from the STAC search expire after ~45 min.
    # For large ROIs the gap between search and Dask fetching pixels
    # can exceed this, causing: RasterioIOError "not a supported format"
    # Fix: re-sign every item right before stackstac so tokens are fresh.
    items = [pc.sign(item) for item in items]

    # ── 4. Stack ────────────────────────────────────────────────────────
    # rescale=False: keep raw DN values (0-10000); we scale in calculate_indices.
    # dtype=float:   avoids the safe-cast ValueError with rescale=False.
    # epsg=3857:     Web Mercator — resolution unit is METRES not degrees.
    # bounds_latlon: pass lon/lat bbox; stackstac reprojects to 3857 internally.
    stack = stackstac.stack(
        items,
        assets        = ['B04', 'B08', 'B11', 'SCL'],
        resolution    = resolution,
        epsg          = TARGET_EPSG,
        bounds_latlon = bbox,
        dtype         = float,
        rescale       = False,
        chunksize     = 1024,    # Dask chunk size — keeps memory predictable
    )

    # stackstac returns a DataArray with band as a dimension
    # Convert to Dataset so bands are named variables (ds['B04'] etc.)
    ds = stack.to_dataset(dim='band')

    # ── 4. Cloud mask ───────────────────────────────────────────────────
    ds = mask_clouds_scl(ds)

    # ── 5. Indices ──────────────────────────────────────────────────────
    ds = calculate_indices(ds)

    # ── 6. Composite ────────────────────────────────────────────────────
    composites = build_composites(ds)

    # ── 7. Velocity ─────────────────────────────────────────────────────
    composites = calculate_velocity(composites)

    # ── 8. Trim warm-up ─────────────────────────────────────────────────
    return composites.sel(time=slice(start_date, end_date))


# ---------------------------------------------------------------------------
# SPATIAL SUMMARY
# ---------------------------------------------------------------------------

def _compute_with_retry(da, retries=4, backoff=10):
    """Retry dask .compute() on intermittent Azure blob read failures."""
    import time
    for attempt in range(retries):
        try:
            return da.compute()
        except Exception as e:
            err = str(e)
            retriable = any(k in err for k in [
                'Read failed', 'RasterioIOError', 'RuntimeError',
                'not recognized as', 'CPLE_OpenFailed'
            ])
            if retriable and attempt < retries - 1:
                wait = backoff * (2 ** attempt)  # 10s, 20s, 40s
                print(f"    [RETRY {attempt+1}/{retries-1}] Network error, "
                      f"retrying in {wait}s: {err[:80]}")
                time.sleep(wait)
            else:
                raise
    return da.compute()


def spatial_mean(
    composites: xr.Dataset,
    variables:  list = None
) -> pd.DataFrame:
    """
    Spatial mean over x/y → one row per composite timestep.
    Triggers Dask computation (data actually downloads here).
    Retries up to 4x on intermittent Azure blob read failures.

    Returns pd.DataFrame with columns: time, NDVI, LSWI, NDVI_velocity
    """
    if variables is None:
        variables = ['NDVI', 'LSWI', 'NDVI_velocity']

    series_list = []
    for var in variables:
        if var in composites:
            s = _compute_with_retry(composites[var].mean(dim=['x', 'y']))
            series_list.append(s.to_series().rename(var))

    if not series_list:
        return pd.DataFrame(columns=['time'] + variables)

    df = pd.concat(series_list, axis=1).reset_index()
    df = df.rename(columns={'time': 'time'})
    return df
