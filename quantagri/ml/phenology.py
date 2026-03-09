"""
Phenology extraction using NDVI changepoint detection.
Identifies green-up, peak, and senescence dates per season.
"""
import numpy as np
import pandas as pd


def _pelt_changepoints(series: np.ndarray, penalty: float = 3.0) -> list:
    """
    Lightweight PELT-style changepoint detection.
    Falls back to simple derivative if ruptures not installed.
    """
    try:
        import ruptures as rpt
        algo = rpt.Pelt(model="rbf").fit(series.reshape(-1, 1))
        return algo.predict(pen=penalty)[:-1]  # drop final index
    except ImportError:
        # Fallback: find points where derivative changes sign
        diff = np.diff(series)
        sign_changes = np.where(np.diff(np.sign(diff)))[0] + 1
        return sign_changes.tolist()


class PhenologyExtractor:
    """
    Extracts crop phenology stages from monthly NDVI time series.
    """

    def extract(self, monthly_df: pd.DataFrame) -> dict:
        """
        monthly_df: rows for one commodity+region+season_year,
                    sorted by month, with ndvi_mean column.
        Returns dict of phenology metrics.
        """
        ndvi = monthly_df.sort_values("month")["ndvi_mean"].dropna().values
        months = monthly_df.sort_values("month")["month"].values

        if len(ndvi) < 3:
            return self._empty()

        peak_idx = int(np.argmax(ndvi))
        peak_ndvi = float(ndvi[peak_idx])
        peak_month = int(months[peak_idx]) if peak_idx < len(months) else -1

        green_up_ndvi = float(ndvi[0])
        green_up_month = int(months[0])

        # Senescence: steepest post-peak decline
        post_peak = ndvi[peak_idx:]
        if len(post_peak) > 1:
            decline_rate = float((post_peak[-1] - post_peak[0]) / (len(post_peak) - 1))
            senes_month = int(months[peak_idx + int(np.argmin(np.diff(post_peak))) + 1]) \
                if peak_idx + 1 < len(months) else -1
        else:
            decline_rate = 0.0
            senes_month = -1

        # Changepoints
        cps = _pelt_changepoints(ndvi)

        # Season length proxy
        days_green = (peak_month - green_up_month) * 30  # rough

        return {
            "green_up_month": green_up_month,
            "green_up_ndvi": green_up_ndvi,
            "peak_month": peak_month,
            "peak_ndvi": peak_ndvi,
            "senescence_month": senes_month,
            "decline_rate": decline_rate,
            "days_to_peak": days_green,
            "n_changepoints": len(cps),
            "season_length_months": len(ndvi),
        }

    def _empty(self) -> dict:
        return {k: np.nan for k in [
            "green_up_month", "green_up_ndvi", "peak_month", "peak_ndvi",
            "senescence_month", "decline_rate", "days_to_peak",
            "n_changepoints", "season_length_months",
        ]}

    def batch_extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run extraction for all commodity+region+season_year combos in df."""
        rows = []
        for (commodity, region, yr), grp in df.groupby(["commodity", "region_id", "season_year"]):
            result = self.extract(grp)
            result.update({"commodity": commodity, "region_id": region, "season_year": yr})
            rows.append(result)
        return pd.DataFrame(rows)
