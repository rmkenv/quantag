"""
QUANTAGRI COMMODITY CONFIGURATION
----------------------------------
Growing season windows and regional geometries for all commodity-region pairs.

Split regions:
    - Texas cotton: South Texas vs Texas Panhandle (15-degree N-S span,
      ~2-3 week planting offset between zones)
    - Xinjiang cotton: Tarim Basin (south) vs Northern Oasis Belt,
      materially different thermal regimes and emergence timing.

Sources:
    USDA NASS Crop Progress (US), CONAB Safra (Brazil),
    FAO GAEZ v4, ICRISAT, Chinese Ministry of Agriculture / ICAC.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd


# ---------------------------------------------------------------------------
# DATA STRUCTURE
# ---------------------------------------------------------------------------

@dataclass
class GrowingSeason:
    """
    Single growing season configuration for one commodity-region.

    Attributes:
        label        : Human-readable name.
        commodity    : Commodity key ('corn','soy','wheat','cotton','sugar').
        region_id    : Unique region slug.
        geometry     : (min_lat, min_lon, max_lat, max_lon) bounding box.
        start_month  : Month of emergence / green-up (1–12).
        end_month    : Month of physiological maturity (1–12).
                       end_month < start_month → season crosses calendar year.
        hemisphere   : 'N' or 'S'.
        notes        : Methodological rationale.
    """
    label        : str
    commodity    : str
    region_id    : str
    geometry     : Tuple[float, float, float, float]
    start_month  : int
    end_month    : int
    hemisphere   : str = 'N'
    notes        : str = ''

    def crosses_year(self) -> bool:
        return self.end_month < self.start_month

    def to_ee_geometry(self):
        raise NotImplementedError("GEE not used in PC version. Use season.geometry tuple directly.")


# ---------------------------------------------------------------------------
# COMMODITY SEASON REGISTRY
# ---------------------------------------------------------------------------

COMMODITY_SEASONS: List[GrowingSeason] = [

    # -----------------------------------------------------------------------
    # CORN
    # -----------------------------------------------------------------------
    GrowingSeason(
        label='Corn — Iowa (US Corn Belt)',
        commodity='corn', region_id='iowa_us',
        geometry=(40.5, -96.7, 43.5, -90.1),
        start_month=5, end_month=9, hemisphere='N',
        notes=(
            'USDA NASS: Iowa emergence ~wk3 May; black layer maturity ~wk3 Sep. '
            'Harvest Oct–Nov excluded to avoid senescence-litter NDVI contamination.'
        )
    ),
    GrowingSeason(
        label='Corn — Illinois (US Corn Belt)',
        commodity='corn', region_id='illinois_us',
        geometry=(37.0, -91.5, 42.5, -87.5),
        start_month=5, end_month=9, hemisphere='N',
        notes=(
            'Southern Illinois emergence ~wk2 May. Slight southward shift '
            'allows earlier planting vs Iowa.'
        )
    ),

    # -----------------------------------------------------------------------
    # SOYBEANS
    # -----------------------------------------------------------------------
    GrowingSeason(
        label='Soy — Mato Grosso Principal (Safra)',
        commodity='soy', region_id='mato_grosso_br',
        geometry=(-18.0, -61.0, -7.0, -50.0),
        start_month=10, end_month=3, hemisphere='S',
        notes=(
            'CONAB Safra: planting Oct–Nov, peak vegetative growth Dec–Jan, '
            'harvest Feb–Mar. Crosses calendar year. Safrinha corn not tracked here.'
        )
    ),
    GrowingSeason(
        label='Soy — Illinois (US)',
        commodity='soy', region_id='illinois_us',
        geometry=(37.0, -91.5, 42.5, -87.5),
        start_month=5, end_month=10, hemisphere='N',
        notes=(
            'USDA NASS: Illinois soy emergence ~wk4 May; R7 maturity ~wk3 Sep; '
            'harvest window Oct included through pod-shatter stage.'
        )
    ),

    # -----------------------------------------------------------------------
    # WHEAT
    # -----------------------------------------------------------------------
    GrowingSeason(
        label='Wheat — Kansas HRW (US)',
        commodity='wheat', region_id='kansas_us',
        geometry=(37.0, -102.0, 40.0, -94.6),
        start_month=3, end_month=6, hemisphere='N',
        notes=(
            'HRW: planted Sep–Oct, dormancy Nov–Feb excluded. Green-up Mar; '
            'heading Apr–May; physiological maturity Jun. USDA NASS Kansas.'
        )
    ),
    GrowingSeason(
        label='Wheat — Rostov Oblast (Black Sea)',
        commodity='wheat', region_id='rostov_ru',
        geometry=(45.8, 38.2, 50.2, 44.3),
        start_month=3, end_month=7, hemisphere='N',
        notes=(
            'Winter wheat dominant. FAO GAEZ v4: green-up Mar–Apr post snow-melt; '
            'heading May–Jun; harvest Jun–Jul.'
        )
    ),

    # -----------------------------------------------------------------------
    # COTTON — SPLIT REGIONS
    # -----------------------------------------------------------------------

    # Texas: split into South Texas and Panhandle (~2-3 week emergence offset)
    GrowingSeason(
        label='Cotton — South Texas (US)',
        commodity='cotton', region_id='south_texas_us',
        geometry=(25.8, -106.6, 31.5, -93.5),
        start_month=4, end_month=9, hemisphere='N',
        notes=(
            'Lower Rio Grande Valley and coastal plains: earlier thermal '
            'accumulation allows April emergence. USDA NASS: planting Mar–Apr, '
            'boll opening Aug–Sep. Defoliation/harvest Oct excluded.'
        )
    ),
    GrowingSeason(
        label='Cotton — Texas Panhandle & Rolling Plains (US)',
        commodity='cotton', region_id='north_texas_us',
        geometry=(31.5, -106.6, 36.5, -93.5),
        start_month=5, end_month=10, hemisphere='N',
        notes=(
            'Lubbock and High Plains: later last-frost date pushes emergence '
            'to mid-May. Boll opening Sep–Oct. Frost risk from Nov onward '
            'constrains season end. USDA NASS Texas Crop Progress.'
        )
    ),

    # Xinjiang: split into Tarim Basin (south) and Northern Oasis Belt
    GrowingSeason(
        label='Cotton — Xinjiang Tarim Basin (China)',
        commodity='cotton', region_id='xinjiang_tarim_cn',
        geometry=(34.3, 73.5, 41.5, 96.4),
        start_month=4, end_month=9, hemisphere='N',
        notes=(
            'Tarim Basin oasis agriculture: hot, arid continental climate. '
            'Earlier emergence (April) due to higher GDD accumulation. '
            'Chinese MoA / ICAC: planting Apr, boll opening Aug–Sep. '
            'Cut at Sep to avoid frost-driven senescence in northern sub-areas.'
        )
    ),
    GrowingSeason(
        label='Cotton — Xinjiang Northern Oasis Belt (China)',
        commodity='cotton', region_id='xinjiang_north_cn',
        geometry=(41.5, 73.5, 49.2, 96.4),
        start_month=5, end_month=9, hemisphere='N',
        notes=(
            'Northern Xinjiang: Turpan Depression and Ili Valley belt. '
            'Later frost-free date delays planting to May. Frost risk from '
            'Oct limits season. Mechanisation expanding but harvest excluded.'
        )
    ),

    # -----------------------------------------------------------------------
    # SUGAR
    # -----------------------------------------------------------------------
    GrowingSeason(
        label='Sugar Cane — Sao Paulo (Brazil)',
        commodity='sugar', region_id='sao_paulo_br',
        geometry=(-25.3, -53.1, -19.8, -44.2),
        start_month=10, end_month=4, hemisphere='S',
        notes=(
            'CONAB / UNICA: wet-season growth Oct–Apr captures ratoon regrowth '
            'and new plant crops. Dry-season harvest May–Sep excluded: '
            'mechanical harvesting and trash blanketing confound NDVI. '
            'Crosses calendar year boundary.'
        )
    ),
    GrowingSeason(
        label='Sugar Cane — Uttar Pradesh (India)',
        commodity='sugar', region_id='uttar_pradesh_in',
        geometry=(23.8, 77.0, 30.4, 84.6),
        start_month=2, end_month=11, hemisphere='N',
        notes=(
            'Spring plant Feb–Mar; autumn plant Sep–Oct. Harvest Dec–Apr excluded. '
            'ICRISAT / ISMA. Long season (~10 months) captures monsoon-driven '
            'growth acceleration Jul–Sep as a distinct velocity event.'
        )
    ),
]


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def get_season_date_range(
    season: GrowingSeason,
    analysis_year: int
) -> Tuple[str, str]:
    """
    Return (start_str, end_str) ISO date strings for a season-year.
    Handles year-crossing seasons (end_month < start_month).
    Uses pandas — no GEE dependency.
    """
    import pandas as pd
    start    = pd.Timestamp(year=analysis_year, month=season.start_month, day=1)
    end_year = analysis_year + 1 if season.crosses_year() else analysis_year
    end      = (
        pd.Timestamp(year=end_year, month=season.end_month, day=1)
        + pd.offsets.MonthEnd(1)
    )
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def get_season(commodity: str, region_id: str) -> GrowingSeason:
    """Retrieve a single GrowingSeason by commodity + region_id."""
    matches = [
        s for s in COMMODITY_SEASONS
        if s.commodity == commodity and s.region_id == region_id
    ]
    if not matches:
        available = [(s.commodity, s.region_id) for s in COMMODITY_SEASONS]
        raise ValueError(
            f"No config for '{commodity}'/'{region_id}'. Available: {available}"
        )
    return matches[0]


def list_seasons() -> None:
    """Print a formatted summary of all configured seasons."""
    print(f"\n{'COMMODITY':<10} {'REGION':<28} {'WINDOW':<22} {'YEAR WRAP'}")
    print('-' * 75)
    for s in COMMODITY_SEASONS:
        w = f'M{s.start_month:02d} → M{s.end_month:02d}'
        wrap = '✓' if s.crosses_year() else ''
        print(f'{s.commodity:<10} {s.region_id:<28} {w:<22} {wrap}')
