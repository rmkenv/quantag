"""
QUANTAGRI YIELD UPDATER
-----------------------
Fetches latest USDA WASDE and CONAB forecasts from public APIs/sources
and updates official_yields.csv automatically.

Sources:
  - USDA PSD API  (free, no key needed): apps.fas.usda.gov/psdonline/app/index.html
  - CONAB API     (free, no key needed): consultasconab.com/api
  - IGC           (scraped)             for Rostov/Russia wheat

Usage:
    python quantagri_yields_updater.py --yields_csv official_yields.csv
    python quantagri_yields_updater.py --yields_csv official_yields.csv --dry_run
"""

import os
import json
import argparse
import requests
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# USDA PSD API CONFIG
# Maps our region_id → USDA commodity + country codes
# Docs: https://apps.fas.usda.gov/psdonline/app/index.html#/app/compositeViz
# ---------------------------------------------------------------------------

USDA_PSD_BASE = "https://apps.fas.usda.gov/psdonline/api/v1/data"

# (commodity_code, country_code, attribute_id, unit_conversion)
# attribute_id 20 = Production, attribute_id 28 = Yield
# Units: USDA reports in 1000 MT for production, MT/HA for yield
USDA_TARGETS = {
    # corn
    ('corn', 'iowa_us'):        ('0440000', '2900000', 28, 1.0),   # Corn, US yield MT/HA → we convert to BU/AC
    ('corn', 'illinois_us'):    ('0440000', '2900000', 28, 1.0),
    # soy
    ('soy',  'illinois_us'):    ('2222000', '2900000', 28, 1.0),
    # wheat
    ('wheat','kansas_us'):      ('0410000', '2900000', 28, 1.0),
    # cotton  
    ('cotton','south_texas_us'):('0813100', '2900000', 28, 1.0),
    ('cotton','north_texas_us'):('0813100', '2900000', 28, 1.0),
}

# MT/HA → BU/AC conversion factors
MT_HA_TO_BU_AC = {
    'corn':   15.926,   # 1 MT/HA = 15.926 BU/AC
    'soy':    14.870,   # 1 MT/HA = 14.870 BU/AC  
    'wheat':  14.870,
    'cotton': None,     # cotton in LBS/AC — different conversion
}

# ---------------------------------------------------------------------------
# CONAB API CONFIG  (Brazilian soy/sugar)
# ---------------------------------------------------------------------------

CONAB_BASE = "https://consultasconab.com/api"

# ---------------------------------------------------------------------------
# FETCH FUNCTIONS
# ---------------------------------------------------------------------------

def fetch_usda_psd(commodity_code: str, country_code: str, attribute_id: int) -> dict:
    """
    Fetch time series from USDA PSD API.
    Returns dict of {year: value}.
    """
    url = f"{USDA_PSD_BASE}/commodity/{commodity_code}/country/{country_code}/year/0/attribute/{attribute_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        result = {}
        for row in data:
            yr  = row.get('marketYear') or row.get('year')
            val = row.get('value')
            if yr and val is not None:
                result[int(yr)] = float(val)
        return result
    except Exception as e:
        print(f"  [USDA API ERROR] {commodity_code}/{country_code}: {e}")
        return {}


def fetch_brazil_soy_conab() -> dict:
    """
    Fetch Brazilian soy production from CONAB.
    Returns dict of {year: yield_sacks_per_ha} for Mato Grosso.
    Falls back to USDA PSD Brazil soy if CONAB unavailable.
    """
    # Try CONAB first
    try:
        url = f"{CONAB_BASE}/v1/previsoes?cultura=soja&estado=MT"
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            result = {}
            for row in data.get('data', []):
                yr  = row.get('safra', '')[:4]
                val = row.get('produtividade_sacas_ha')
                if yr and val:
                    result[int(yr)] = float(val)
            if result:
                print(f"  [CONAB] Fetched {len(result)} years of Mato Grosso soy")
                return result
    except Exception:
        pass

    # Fallback: USDA PSD Brazil soy yield (MT/HA → sacks/HA, 1 MT = 16.67 sacks)
    print("  [CONAB] Falling back to USDA PSD Brazil soy")
    raw = fetch_usda_psd('2222000', '0351000', 28)  # Brazil country code
    return {yr: round(val * 16.67, 1) for yr, val in raw.items()} if raw else {}


def fetch_russia_wheat_igc() -> dict:
    """
    Fetch Russia wheat yield from USDA PSD (best available free source).
    Returns dict of {year: yield_t_ha}.
    """
    # Russia country code in USDA PSD = 3550000
    raw = fetch_usda_psd('0410000', '3550000', 28)
    # USDA reports Russia wheat in MT/HA — that's what we store
    return raw


def fetch_usda_corn_yield_bu_ac(country_code: str = '2900000') -> dict:
    """US corn yield in BU/AC from USDA PSD."""
    raw = fetch_usda_psd('0440000', country_code, 28)
    # Convert MT/HA → BU/AC
    factor = MT_HA_TO_BU_AC['corn']
    return {yr: round(val * factor, 1) for yr, val in raw.items()}


def fetch_usda_soy_yield_bu_ac(country_code: str = '2900000') -> dict:
    """US soy yield in BU/AC from USDA PSD."""
    raw = fetch_usda_psd('2222000', country_code, 28)
    factor = MT_HA_TO_BU_AC['soy']
    return {yr: round(val * factor, 1) for yr, val in raw.items()}


def fetch_usda_wheat_yield_bu_ac(country_code: str = '2900000') -> dict:
    """US wheat yield in BU/AC from USDA PSD."""
    raw = fetch_usda_psd('0410000', country_code, 28)
    factor = MT_HA_TO_BU_AC['wheat']
    return {yr: round(val * factor, 1) for yr, val in raw.items()}


def fetch_usda_cotton_yield_lbs_ac(country_code: str = '2900000') -> dict:
    """US cotton yield in LBS/AC from USDA PSD."""
    raw = fetch_usda_psd('0813100', country_code, 28)
    # USDA cotton in KG/HA → LBS/AC: 1 KG/HA = 0.8922 LBS/AC
    return {yr: round(val * 0.8922, 1) for yr, val in raw.items()}


# ---------------------------------------------------------------------------
# MAIN UPDATE LOGIC
# ---------------------------------------------------------------------------

FETCH_MAP = {
    ('corn',   'iowa_us'):           fetch_usda_corn_yield_bu_ac,
    ('corn',   'illinois_us'):       fetch_usda_corn_yield_bu_ac,
    ('soy',    'mato_grosso_br'):    fetch_brazil_soy_conab,
    ('soy',    'illinois_us'):       fetch_usda_soy_yield_bu_ac,
    ('wheat',  'kansas_us'):         fetch_usda_wheat_yield_bu_ac,
    ('wheat',  'rostov_ru'):         fetch_russia_wheat_igc,
    ('cotton', 'south_texas_us'):    fetch_usda_cotton_yield_lbs_ac,
    ('cotton', 'north_texas_us'):    fetch_usda_cotton_yield_lbs_ac,
    ('cotton', 'xinjiang_tarim_cn'): None,   # No free API — update manually
    ('cotton', 'xinjiang_north_cn'): None,
    ('sugar',  'sao_paulo_br'):      None,   # CONAB sugar — update manually
    ('sugar',  'uttar_pradesh_in'):  None,   # Indian sugar — update manually
}

MIN_YEAR = 2016
FORECAST_YEARS = 2   # Add forecast rows this many years into the future


def update_yields(yields_csv: str, dry_run: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"  QuantAgri Yield Updater — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  {'DRY RUN — no changes written' if dry_run else 'LIVE RUN'}")
    print(f"{'='*60}\n")

    df = pd.read_csv(yields_csv)
    current_year = datetime.utcnow().year
    updates = 0
    skipped = 0

    for (commodity, region_id), fetch_fn in FETCH_MAP.items():
        print(f"\n--- {commodity} / {region_id} ---")

        if fetch_fn is None:
            print(f"  [SKIP] No API available — update manually from CONAB/USDA/IGC")
            skipped += 1
            continue

        fetched = fetch_fn()
        if not fetched:
            print(f"  [EMPTY] No data returned from API")
            continue

        for year in range(MIN_YEAR, current_year + FORECAST_YEARS + 1):
            api_val = fetched.get(year)

            # Find existing row
            mask = (
                (df['commodity']  == commodity) &
                (df['region_id']  == region_id) &
                (df['year']       == year)
            )
            existing = df[mask]

            if api_val is None:
                if existing.empty:
                    # Add placeholder row for future years
                    new_row = pd.DataFrame([{
                        'commodity':      commodity,
                        'region_id':      region_id,
                        'year':           year,
                        'official_yield': 0.0
                    }])
                    df = pd.concat([df, new_row], ignore_index=True)
                    print(f"  {year}: added placeholder 0.0 (no API data yet)")
                    updates += 1
                continue

            if existing.empty:
                # Insert new row
                new_row = pd.DataFrame([{
                    'commodity':      commodity,
                    'region_id':      region_id,
                    'year':           year,
                    'official_yield': api_val
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                print(f"  {year}: ADDED {api_val}")
                updates += 1
            else:
                old_val = float(existing['official_yield'].iloc[0])
                if old_val == 0.0 or abs(old_val - api_val) > 0.1:
                    df.loc[mask, 'official_yield'] = api_val
                    print(f"  {year}: UPDATED {old_val} → {api_val}")
                    updates += 1
                else:
                    print(f"  {year}: unchanged ({old_val})")

    # Sort and save
    df = df.sort_values(['commodity', 'region_id', 'year']).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"  {updates} updates, {skipped} skipped (manual)")
    if not dry_run:
        df.to_csv(yields_csv, index=False)
        print(f"  Saved → {yields_csv}")
    else:
        print(f"  DRY RUN — nothing written")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='QuantAgri Yield Updater')
    p.add_argument('--yields_csv', default='official_yields.csv',
                   help='Path to official_yields.csv')
    p.add_argument('--dry_run', action='store_true',
                   help='Print changes without writing to file')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    update_yields(args.yields_csv, dry_run=args.dry_run)
