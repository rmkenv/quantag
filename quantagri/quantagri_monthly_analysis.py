"""
QUANTAGRI MONTHLY ANALYSIS
---------------------------
Aggregates daily live results into monthly summaries and runs
statistical analysis: z-scores, anomaly detection, trend tests,
year-over-year comparisons, and yield correlation.

Reads:  quantagri_live/quantagri_live_results.csv
        quantagri_historical/quantagri_monthly_ALL.csv  (optional baseline)
        official_yields.csv

Writes: quantagri_analysis/quantagri_monthly_summary.csv
        quantagri_analysis/quantagri_anomaly_report.csv
        quantagri_analysis/quantagri_stats_report.txt

Usage:
    python quantagri_monthly_analysis.py
    python quantagri_monthly_analysis.py --live_csv quantagri_live/quantagri_live_results.csv
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

warnings.filterwarnings('ignore')

DEFAULT_LIVE_CSV       = 'quantagri_live/quantagri_live_results.csv'
DEFAULT_HISTORICAL_CSV = 'quantagri_historical/quantagri_monthly_ALL.csv'
DEFAULT_YIELDS_CSV     = 'official_yields.csv'
DEFAULT_OUTPUT_DIR     = 'quantagri_analysis'

Z_ALERT_THRESHOLD      = 1.5   # flag if current NDVI z-score beyond ±1.5σ


# ---------------------------------------------------------------------------
# STEP 1 — Aggregate daily live results to monthly
# ---------------------------------------------------------------------------

def aggregate_live_to_monthly(live_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group daily live results by commodity + region + year + month.
    Takes the LAST observation of the month as the representative value
    (most data), plus mean/std across the month's observations.
    """
    live_df = live_df.copy()
    live_df['as_of_date'] = pd.to_datetime(live_df['as_of_date'])
    live_df['year']  = live_df['as_of_date'].dt.year
    live_df['month'] = live_df['as_of_date'].dt.month

    agg_rows = []
    group_cols = ['commodity', 'region_id', 'year', 'month']

    for keys, grp in live_df.groupby(group_cols):
        commodity, region_id, year, month = keys
        grp = grp.sort_values('as_of_date')
        last = grp.iloc[-1]

        row = {
            'commodity':            commodity,
            'region_id':            region_id,
            'year':                 year,
            'month':                month,
            'month_label':          pd.Timestamp(year=year, month=month, day=1).strftime('%b %Y'),
            'n_observations':       len(grp),
            'season_pct_elapsed':   last['season_pct_elapsed'],

            # NDVI — end of month value + intra-month stats
            'ndvi_eom':             last['current_ndvi'],
            'ndvi_month_mean':      grp['current_ndvi'].mean(),
            'ndvi_month_std':       grp['current_ndvi'].std(),
            'ndvi_month_min':       grp['current_ndvi'].min(),
            'ndvi_month_max':       grp['current_ndvi'].max(),

            # LSWI
            'lswi_eom':             last['current_lswi'],
            'lswi_month_mean':      grp['current_lswi'].mean(),

            # Velocity
            'velocity_eom':         last['current_ndvi_velocity'],
            'velocity_month_mean':  grp['current_ndvi_velocity'].mean(),

            # Season peaks
            'peak_ndvi':            last['peak_ndvi'],
            'peak_ndvi_date':       last['peak_ndvi_date'],

            # Yield signal
            'yield_surprise':       last['yield_surprise'],
            'surprise_pct':         last['surprise_pct'],
            'calibration_r2':       last['calibration_r2'],

            # Tercile means (season context)
            'tercile_mean_early':   last['tercile_mean_early'],
            'tercile_mean_mid':     last['tercile_mean_mid'],
            'tercile_mean_late':    last['tercile_mean_late'],
        }
        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


# ---------------------------------------------------------------------------
# STEP 2 — Z-score vs historical baseline
# ---------------------------------------------------------------------------

def compute_zscore_vs_historical(
    monthly_live: pd.DataFrame,
    historical:   pd.DataFrame
) -> pd.DataFrame:
    """
    For each live monthly row, compute z-score of current NDVI
    vs the same commodity+region+month across all historical years.
    """
    if historical is None or historical.empty:
        monthly_live['ndvi_zscore']      = np.nan
        monthly_live['ndvi_pct_rank']    = np.nan
        monthly_live['hist_ndvi_mean']   = np.nan
        monthly_live['hist_ndvi_std']    = np.nan
        monthly_live['hist_n_years']     = 0
        return monthly_live

    results = []
    for _, row in monthly_live.iterrows():
        hist = historical[
            (historical['commodity']  == row['commodity']) &
            (historical['region_id']  == row['region_id']) &
            (historical['month']      == row['month'])
        ]['ndvi_mean'].dropna()

        if len(hist) < 3:
            results.append({**row, 'ndvi_zscore': np.nan, 'ndvi_pct_rank': np.nan,
                             'hist_ndvi_mean': np.nan, 'hist_ndvi_std': np.nan,
                             'hist_n_years': len(hist)})
            continue

        mean = hist.mean()
        std  = hist.std()
        z    = (row['ndvi_eom'] - mean) / std if std > 0 else 0.0
        pct  = float(stats.percentileofscore(hist, row['ndvi_eom']))

        results.append({
            **row,
            'ndvi_zscore':    round(z, 3),
            'ndvi_pct_rank':  round(pct, 1),
            'hist_ndvi_mean': round(mean, 4),
            'hist_ndvi_std':  round(std, 4),
            'hist_n_years':   len(hist),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# STEP 3 — Statistical tests
# ---------------------------------------------------------------------------

def run_trend_test(series: pd.Series) -> dict:
    """Mann-Kendall trend test on a time series."""
    if len(series) < 4:
        return {'trend': 'insufficient data', 'tau': np.nan, 'p_value': np.nan}
    try:
        result = stats.kendalltau(range(len(series)), series)
        tau    = result.statistic
        p      = result.pvalue
        trend  = 'increasing' if tau > 0 and p < 0.1 else \
                 'decreasing' if tau < 0 and p < 0.1 else 'no trend'
        return {'trend': trend, 'tau': round(tau, 3), 'p_value': round(p, 3)}
    except Exception:
        return {'trend': 'error', 'tau': np.nan, 'p_value': np.nan}


def compute_yield_correlation(
    monthly_live: pd.DataFrame,
    yields_df:    pd.DataFrame
) -> pd.DataFrame:
    """
    Correlate end-of-season NDVI (season_pct_elapsed > 80%) with official yields.
    """
    rows = []
    for (commodity, region_id), grp in monthly_live.groupby(['commodity', 'region_id']):
        late_season = grp[grp['season_pct_elapsed'] > 80].copy()
        if late_season.empty:
            continue

        yield_data = yields_df[
            (yields_df['commodity'] == commodity) &
            (yields_df['region_id'] == region_id) &
            (yields_df['official_yield'] > 0)
        ].set_index('year')['official_yield']

        merged = late_season.set_index('year')['ndvi_eom'].to_frame().join(
            yield_data.rename('official_yield'), how='inner'
        ).dropna()

        if len(merged) < 3:
            continue

        r, p = stats.pearsonr(merged['ndvi_eom'], merged['official_yield'])
        rows.append({
            'commodity':  commodity,
            'region_id':  region_id,
            'r':          round(r, 3),
            'r2':         round(r**2, 3),
            'p_value':    round(p, 3),
            'n_years':    len(merged),
            'significant': p < 0.1,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# STEP 4 — Anomaly report
# ---------------------------------------------------------------------------

def build_anomaly_report(monthly_with_z: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where z-score exceeds threshold."""
    if 'ndvi_zscore' not in monthly_with_z.columns:
        return pd.DataFrame()

    anomalies = monthly_with_z[
        monthly_with_z['ndvi_zscore'].abs() >= Z_ALERT_THRESHOLD
    ].copy()

    if anomalies.empty:
        return anomalies

    anomalies['direction'] = anomalies['ndvi_zscore'].apply(
        lambda z: 'ABOVE NORMAL' if z > 0 else 'BELOW NORMAL'
    )
    anomalies['severity'] = anomalies['ndvi_zscore'].abs().apply(
        lambda z: 'EXTREME' if z >= 2.5 else 'SIGNIFICANT' if z >= 2.0 else 'MODERATE'
    )
    return anomalies.sort_values('ndvi_zscore', key=abs, ascending=False)


# ---------------------------------------------------------------------------
# STEP 5 — Text report
# ---------------------------------------------------------------------------

def write_text_report(
    monthly:     pd.DataFrame,
    anomalies:   pd.DataFrame,
    correlations: pd.DataFrame,
    output_path: str
) -> None:
    lines = []
    lines.append("=" * 65)
    lines.append("  QUANTAGRI MONTHLY STATISTICAL REPORT")
    lines.append(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("=" * 65)

    # --- Summary by commodity/region ---
    lines.append("\n[1] MONTHLY NDVI SUMMARY (latest month per region)\n")
    latest = monthly.sort_values('month_label').groupby(
        ['commodity', 'region_id']
    ).last().reset_index()

    for _, row in latest.iterrows():
        z_str = f"z={row.get('ndvi_zscore', 'n/a'):.2f}" \
                if pd.notna(row.get('ndvi_zscore')) else "z=n/a"
        pct_str = f"pct={row.get('ndvi_pct_rank', 'n/a'):.0f}th" \
                  if pd.notna(row.get('ndvi_pct_rank')) else ""
        lines.append(
            f"  {row['commodity']:<8} {row['region_id']:<22} "
            f"{row['month_label']:<10}  "
            f"NDVI={row['ndvi_eom']:.4f}  {z_str}  {pct_str}"
        )

    # --- Anomalies ---
    lines.append("\n[2] ANOMALIES (|z| >= {:.1f}σ)\n".format(Z_ALERT_THRESHOLD))
    if anomalies.empty:
        lines.append("  No anomalies detected.")
    else:
        for _, row in anomalies.iterrows():
            lines.append(
                f"  ⚠️  {row['commodity']}/{row['region_id']} "
                f"{row['month_label']}  "
                f"z={row['ndvi_zscore']:.2f}  "
                f"{row['severity']} {row['direction']}"
            )

    # --- Yield correlations ---
    lines.append("\n[3] NDVI → YIELD CORRELATIONS\n")
    if correlations.empty:
        lines.append("  Insufficient data for correlation (need 3+ years).")
    else:
        for _, row in correlations.iterrows():
            sig = "✓ significant" if row['significant'] else ""
            lines.append(
                f"  {row['commodity']:<8} {row['region_id']:<22} "
                f"r={row['r']:.3f}  R²={row['r2']:.3f}  "
                f"p={row['p_value']:.3f}  n={row['n_years']}  {sig}"
            )

    # --- Velocity trend ---
    lines.append("\n[4] NDVI VELOCITY TREND (season direction)\n")
    for (commodity, region_id), grp in monthly.groupby(['commodity', 'region_id']):
        vel = grp.sort_values('month')['velocity_eom'].dropna()
        if len(vel) < 2:
            continue
        trend = run_trend_test(vel)
        latest_vel = vel.iloc[-1]
        arrow = "↑" if latest_vel > 0.005 else "↓" if latest_vel < -0.005 else "→"
        lines.append(
            f"  {commodity:<8} {region_id:<22}  "
            f"velocity={latest_vel:.5f} {arrow}  "
            f"trend={trend['trend']}  τ={trend['tau']}"
        )

    lines.append("\n" + "=" * 65 + "\n")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_analysis(
    live_csv:       str,
    historical_csv: str,
    yields_csv:     str,
    output_dir:     str
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[LOAD] Live results: {live_csv}")
    live_df = pd.read_csv(live_csv)
    print(f"  {len(live_df)} rows loaded")

    historical = None
    if os.path.exists(historical_csv):
        print(f"[LOAD] Historical baseline: {historical_csv}")
        historical = pd.read_csv(historical_csv)
        print(f"  {len(historical)} rows loaded")
    else:
        print(f"[SKIP] No historical baseline found at {historical_csv}")
        print(f"  Z-scores will be skipped. Run historical workflow first.")

    yields_df = pd.DataFrame()
    if os.path.exists(yields_csv):
        yields_df = pd.read_csv(yields_csv)

    # Step 1 — aggregate to monthly
    print("\n[1] Aggregating daily → monthly...")
    monthly = aggregate_live_to_monthly(live_df)
    print(f"  {len(monthly)} monthly rows")

    # Step 2 — z-scores
    print("[2] Computing z-scores vs historical baseline...")
    monthly = compute_zscore_vs_historical(monthly, historical)

    # Step 3 — yield correlation
    print("[3] Computing yield correlations...")
    correlations = compute_yield_correlation(monthly, yields_df)

    # Step 4 — anomalies
    print("[4] Detecting anomalies...")
    anomalies = build_anomaly_report(monthly)
    print(f"  {len(anomalies)} anomalies found")

    # Step 5 — write outputs
    monthly_path     = os.path.join(output_dir, 'quantagri_monthly_summary.csv')
    anomaly_path     = os.path.join(output_dir, 'quantagri_anomaly_report.csv')
    corr_path        = os.path.join(output_dir, 'quantagri_correlations.csv')
    report_path      = os.path.join(output_dir, 'quantagri_stats_report.txt')

    monthly.to_csv(monthly_path, index=False)
    anomalies.to_csv(anomaly_path, index=False)
    correlations.to_csv(corr_path, index=False)
    write_text_report(monthly, anomalies, correlations, report_path)

    print(f"\n[DONE] Output in {output_dir}/")
    print(f"  {monthly_path}")
    print(f"  {anomaly_path}")
    print(f"  {corr_path}")
    print(f"  {report_path}")


def parse_args():
    p = argparse.ArgumentParser(description='QuantAgri Monthly Analysis')
    p.add_argument('--live_csv',       default=DEFAULT_LIVE_CSV)
    p.add_argument('--historical_csv', default=DEFAULT_HISTORICAL_CSV)
    p.add_argument('--yields_csv',     default=DEFAULT_YIELDS_CSV)
    p.add_argument('--output_dir',     default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_analysis(
        live_csv       = args.live_csv,
        historical_csv = args.historical_csv,
        yields_csv     = args.yields_csv,
        output_dir     = args.output_dir
    )
