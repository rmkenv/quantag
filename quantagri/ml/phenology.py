"""
quantagri/ml/phenology.py
Crop phenology extractor — detects green-up, peak, senescence.
"""
import numpy as np
import pandas as pd

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False


class PhenologyExtractor:

    def extract(self, monthly_ndvi: pd.Series, months: pd.Series) -> dict:
        """
        Extract phenology metrics from a season's monthly NDVI series.
        monthly_ndvi: Series of NDVI values
        months:       Corresponding month numbers
        """
        if len(monthly_ndvi) < 3:
            return {}

        ndvi   = monthly_ndvi.values
        mo     = months.values
        peak_i = int(np.argmax(ndvi))

        green_up_month   = int(mo[0])
        peak_month       = int(mo[peak_i])
        peak_ndvi        = float(ndvi[peak_i])
        senescence_month = int(mo[-1])
        decline_rate     = float((peak_ndvi - ndvi[-1]) / max(peak_ndvi, 1e-6))
        days_to_peak     = int(peak_i)

        # Changepoint detection
        n_changepoints = 0
        if HAS_RUPTURES and len(ndvi) >= 4:
            try:
                algo = rpt.Pelt(model="rbf").fit(ndvi.reshape(-1, 1))
                cps  = algo.predict(pen=1.0)
                n_changepoints = max(0, len(cps) - 1)
            except Exception:
                n_changepoints = _derivative_changepoints(ndvi)
        else:
            n_changepoints = _derivative_changepoints(ndvi)

        return {
            "green_up_month":   green_up_month,
            "peak_month":       peak_month,
            "peak_ndvi":        peak_ndvi,
            "senescence_month": senescence_month,
            "decline_rate":     decline_rate,
            "days_to_peak":     days_to_peak,
            "n_changepoints":   n_changepoints,
        }

    def batch_extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract phenology for all commodity/region/season_year combos."""
        rows = []
        for (commodity, region_id, season_year), grp in df.groupby(
            ["commodity", "region_id", "season_year"]
        ):
            grp = grp.sort_values("month")
            ndvi_col = "ndvi_mean" if "ndvi_mean" in grp.columns else "ndvi_mean_avg"
            if ndvi_col not in grp.columns:
                continue
            metrics = self.extract(grp[ndvi_col], grp["month"])
            if metrics:
                rows.append({
                    "commodity":   commodity,
                    "region_id":   region_id,
                    "season_year": int(season_year),
                    **metrics,
                })
        return pd.DataFrame(rows)


def _derivative_changepoints(ndvi: np.ndarray) -> int:
    """Fallback changepoint detection using sign changes in derivative."""
    diff  = np.diff(ndvi)
    signs = np.sign(diff)
    changes = int(np.sum(np.diff(signs) != 0))
    return changes
