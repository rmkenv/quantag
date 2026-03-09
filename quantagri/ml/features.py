"""
quantagri/ml/features.py
Season-level feature engineering from monthly satellite data.
"""
import pandas as pd
import numpy as np

FEAT_COLS = [
    "ndvi_mean_avg",
    "ndvi_max_peak",
    "ndvi_mean_std",
    "lswi_mean_avg",
    "lswi_max_peak",
    "vel_mean_avg",
    "vel_max_peak",
    "ndvi_lswi_ratio",
    "green_up_ndvi",
    "peak_ndvi",
    "decline_rate",
    "n_months",
]


def build_season_features(df: pd.DataFrame, commodity: str, region_id: str) -> pd.DataFrame:
    """
    Aggregate monthly satellite rows into one row per season_year.
    Returns a DataFrame with season_year + FEAT_COLS.
    """
    sub = df[(df.commodity == commodity) & (df.region_id == region_id)].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for season_year, grp in sub.groupby("season_year"):
        grp = grp.sort_values("month")

        ndvi   = grp["ndvi_mean"].dropna()
        ndvi_x = grp["ndvi_max"].dropna()
        lswi   = grp["lswi_mean"].dropna()
        lswi_x = grp["lswi_max"].dropna()
        vel    = grp["velocity_mean"].dropna()
        vel_x  = grp["velocity_max"].dropna()

        if len(ndvi) < 2:
            continue

        # Peak and green-up
        peak_idx    = ndvi.idxmax()
        peak_month  = grp.loc[peak_idx, "month"] if peak_idx in grp.index else ndvi.index[0]
        green_up    = float(ndvi.iloc[0])
        peak_val    = float(ndvi.max())
        end_val     = float(ndvi.iloc[-1])
        decline     = (peak_val - end_val) / max(peak_val, 1e-6)

        row = {
            "season_year":    int(season_year),
            "ndvi_mean_avg":  float(ndvi.mean()),
            "ndvi_max_peak":  float(ndvi_x.max()) if not ndvi_x.empty else peak_val,
            "ndvi_mean_std":  float(ndvi.std()) if len(ndvi) > 1 else 0.0,
            "lswi_mean_avg":  float(lswi.mean()) if not lswi.empty else 0.0,
            "lswi_max_peak":  float(lswi_x.max()) if not lswi_x.empty else 0.0,
            "vel_mean_avg":   float(vel.mean()) if not vel.empty else 0.0,
            "vel_max_peak":   float(vel_x.max()) if not vel_x.empty else 0.0,
            "ndvi_lswi_ratio": float(ndvi.mean() / max(abs(lswi.mean()), 1e-6)) if not lswi.empty else 0.0,
            "green_up_ndvi":  green_up,
            "peak_ndvi":      peak_val,
            "decline_rate":   decline,
            "n_months":       int(len(grp)),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("season_year").reset_index(drop=True)


def add_lagged_features(feats: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Add 1-year lagged versions of key features."""
    if feats.empty or len(feats) < 2:
        return feats

    feats = feats.copy().sort_values("season_year")
    for col in ["ndvi_mean_avg", "lswi_mean_avg", "peak_ndvi", "vel_mean_avg"]:
        if col in feats.columns:
            feats[f"{col}_lag{lag}"] = feats[col].shift(lag)

    return feats.dropna().reset_index(drop=True)


def cross_region_divergence(df: pd.DataFrame, commodity: str,
                             region_a: str, region_b: str) -> pd.DataFrame:
    """Compute NDVI divergence between two regions for the same commodity."""
    fa = build_season_features(df, commodity, region_a)[["season_year", "ndvi_mean_avg"]]
    fb = build_season_features(df, commodity, region_b)[["season_year", "ndvi_mean_avg"]]
    merged = fa.merge(fb, on="season_year", suffixes=("_a", "_b"))
    merged["ndvi_divergence"] = merged["ndvi_mean_avg_a"] - merged["ndvi_mean_avg_b"]
    return merged
