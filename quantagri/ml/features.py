"""
Feature engineering from raw monthly satellite metrics.
Produces season-level features for yield modeling.
"""
import pandas as pd
import numpy as np


FEAT_COLS = [
    "ndvi_mean_avg", "ndvi_max_peak", "ndvi_mean_std",
    "lswi_mean_avg", "lswi_max_peak",
    "vel_mean_avg", "vel_max_peak",
    "ndvi_lswi_ratio", "green_up_ndvi", "peak_ndvi",
    "decline_rate", "n_months",
]


def build_season_features(df: pd.DataFrame, commodity: str, region_id: str) -> pd.DataFrame:
    """
    Aggregate monthly satellite rows into one row per season_year.
    Returns a DataFrame with FEAT_COLS as columns.
    """
    sub = df[(df.commodity == commodity) & (df.region_id == region_id)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values(["season_year", "month"])

    grp = sub.groupby("season_year").agg(
        ndvi_mean_avg=("ndvi_mean", "mean"),
        ndvi_max_peak=("ndvi_max", "max"),
        ndvi_mean_std=("ndvi_mean", "std"),
        lswi_mean_avg=("lswi_mean", "mean"),
        lswi_max_peak=("lswi_max", "max"),
        vel_mean_avg=("velocity_mean", "mean"),
        vel_max_peak=("velocity_max", "max"),
        n_months=("month", "count"),
    ).reset_index()

    # NDVI / LSWI ratio — proxy for water stress
    grp["ndvi_lswi_ratio"] = grp["ndvi_mean_avg"] / (grp["lswi_mean_avg"].abs() + 0.01)

    # Intra-season trajectory features
    traj = _trajectory_features(sub)
    grp = grp.merge(traj, on="season_year", how="left")

    return grp


def _trajectory_features(sub: pd.DataFrame) -> pd.DataFrame:
    """Per-season: NDVI at green-up, at peak, and late-season decline rate."""
    rows = []
    for yr, g in sub.groupby("season_year"):
        g = g.sort_values("month")
        ndvi = g["ndvi_mean"].dropna().values
        if len(ndvi) == 0:
            rows.append({"season_year": yr, "green_up_ndvi": np.nan,
                         "peak_ndvi": np.nan, "decline_rate": np.nan})
            continue
        peak_idx = int(np.argmax(ndvi))
        green_up = ndvi[0]
        peak = ndvi[peak_idx]
        post_peak = ndvi[peak_idx:]
        decline = (post_peak[-1] - post_peak[0]) / max(len(post_peak) - 1, 1)
        rows.append({
            "season_year": yr,
            "green_up_ndvi": green_up,
            "peak_ndvi": peak,
            "decline_rate": decline,
        })
    return pd.DataFrame(rows)


def add_lagged_features(feat_df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Add prior-year NDVI and yield surprise as features."""
    df = feat_df.sort_values("season_year").copy()
    for col in ["ndvi_mean_avg", "ndvi_max_peak", "vel_mean_avg"]:
        if col in df.columns:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def cross_region_divergence(
    feat_df: pd.DataFrame,
    region_a: str,
    region_b: str,
    commodity: str,
) -> pd.Series:
    """
    Rolling correlation between two regions' NDVI.
    Drops in correlation flag unusual divergence.
    """
    a = feat_df[(feat_df.commodity == commodity) & (feat_df.region_id == region_a)]\
        .set_index("season_year")["ndvi_mean_avg"]
    b = feat_df[(feat_df.commodity == commodity) & (feat_df.region_id == region_b)]\
        .set_index("season_year")["ndvi_mean_avg"]
    combined = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(combined) < 3:
        return pd.Series(dtype=float)
    rolling_corr = combined["a"].rolling(3).corr(combined["b"])
    divergence = rolling_corr < (rolling_corr.mean() - 1.5 * rolling_corr.std())
    return divergence
