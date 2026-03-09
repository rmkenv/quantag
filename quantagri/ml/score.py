"""
Signal scorer — combines ML outputs with historical z-scores
into a single conviction score per commodity/region.

Score ranges:
  +3  to +4  → Strong LONG  (high conviction, act)
  +1  to +2  → Mild LONG    (monitor, consider small position)
   0         → Neutral      (no edge)
  -1  to -2  → Mild SHORT
  -3  to -4  → Strong SHORT (high conviction, act)
"""
import pandas as pd
import numpy as np
from pathlib import Path


def compute_signal_score(
    live_csv: str = "quantagri_live/quantagri_live_results.csv",
    analysis_csv: str = "quantagri_analysis/quantagri_monthly_summary.csv",
    model_dir: str = "ml_models",
) -> pd.DataFrame:

    live = pd.read_csv(live_csv)

    # Load z-scores if available
    zscore_available = Path(analysis_csv).exists()
    if zscore_available:
        analysis = pd.read_csv(analysis_csv)

    rows = []

    for (commodity, region), grp in live.groupby(["commodity", "region_id"]):
        # Find date column in live CSV
        live_date_col = next(
            (c for c in ["as_of_date", "date", "year_month", "run_date"]
             if c in grp.columns),
            None
        )
        if live_date_col:
            latest = grp.sort_values(live_date_col).iloc[-1]
        else:
            latest = grp.iloc[-1]

        score = 0
        components = {}

        # ── Component 1: ML signal direction (+/-2) ──────────────────────────
        signal = latest.get("ml_signal", "NEUTRAL")
        prob   = latest.get("ml_beat_probability", 0.5)

        if signal == "LONG":
            score += 2
            components["ml_signal"] = "+2 (LONG)"
        elif signal == "SHORT":
            score -= 2
            components["ml_signal"] = "-2 (SHORT)"
        else:
            components["ml_signal"] = "0 (NEUTRAL)"

        # ── Component 2: ML confidence (+/-1) ────────────────────────────────
        confidence = latest.get("ml_confidence", "low")
        if confidence == "high":
            score += (1 if signal == "LONG" else -1 if signal == "SHORT" else 0)
            components["ml_confidence"] = f"±1 ({confidence})"
        elif confidence == "medium":
            components["ml_confidence"] = f"0 ({confidence} — no bonus)"
        else:
            components["ml_confidence"] = "0 (low)"

        # ── Component 3: Anomaly flag (+1 if flagged, direction-adjusted) ────
        anomaly_score = latest.get("ml_anomaly_score", 0.0)
        if pd.notna(anomaly_score) and anomaly_score < -0.05:
            bonus = 1 if signal in ("LONG", "SHORT") else 0
            score += bonus
            components["anomaly"] = f"+{bonus} (flagged, score={anomaly_score:.3f})"
        else:
            components["anomaly"] = "0 (normal)"

        # ── Component 4: Z-score alignment (+/-1) ────────────────────────────
        if zscore_available:
            # Auto-detect date column in analysis CSV
            date_col = next(
                (c for c in ["as_of_date", "year_month", "month_label", "year", "date"]
                 if c in analysis.columns),
                None
            )
            z_sub = analysis[
                (analysis.commodity == commodity) &
                (analysis.region_id == region)
            ]
            if date_col:
                z_sub = z_sub.sort_values(date_col).tail(1)
            else:
                z_sub = z_sub.tail(1)

            if not z_sub.empty and "ndvi_zscore" in z_sub.columns:
                z = float(z_sub["ndvi_zscore"].iloc[0])
                if signal == "LONG" and z > 1.0:
                    score += 1
                    components["zscore"] = f"+1 (z={z:.2f}, confirms LONG)"
                elif signal == "SHORT" and z < -1.0:
                    score += 1
                    components["zscore"] = f"+1 (z={z:.2f}, confirms SHORT)"
                elif signal == "LONG" and z < -1.0:
                    score -= 1
                    components["zscore"] = f"-1 (z={z:.2f}, CONTRADICTS LONG)"
                elif signal == "SHORT" and z > 1.0:
                    score -= 1
                    components["zscore"] = f"-1 (z={z:.2f}, CONTRADICTS SHORT)"
                else:
                    components["zscore"] = f"0 (z={z:.2f}, inconclusive)"
            else:
                components["zscore"] = "0 (no z-score data yet)"
        else:
            components["zscore"] = "0 (analysis CSV not found)"

        # ── Conviction label ──────────────────────────────────────────────────
        if score >= 3:
            conviction = "STRONG LONG 🟢"
        elif score == 2:
            conviction = "MILD LONG 🟡"
        elif score == 1:
            conviction = "LEAN LONG"
        elif score == 0:
            conviction = "NEUTRAL ⚪"
        elif score == -1:
            conviction = "LEAN SHORT"
        elif score == -2:
            conviction = "MILD SHORT 🟡"
        else:
            conviction = "STRONG SHORT 🔴"

        rows.append({
            "commodity":       commodity,
            "region_id":       region,
            "as_of_date":      latest.get(live_date_col) if live_date_col else None,
            "conviction_score": score,
            "conviction":      conviction,
            "ml_signal":       signal,
            "ml_beat_prob":    round(float(prob), 3),
            "ml_confidence":   confidence,
            "anomaly_score":   round(float(anomaly_score), 3) if pd.notna(anomaly_score) else None,
            "yield_surprise":  latest.get("yield_surprise"),
            **{f"component_{k}": v for k, v in components.items()},
        })

    df = pd.DataFrame(rows).sort_values("conviction_score", key=abs, ascending=False)
    return df


def print_scorecard(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  QUANTAGRI SIGNAL SCORECARD")
    print("=" * 65)
    for _, row in df.iterrows():
        print(f"\n  {row.commodity.upper()} / {row.region_id}")
        print(f"  {'─' * 40}")
        print(f"  Conviction : {row.conviction}  (score: {row.conviction_score:+d} / 4)")
        print(f"  ML Signal  : {row.ml_signal} | prob={row.ml_beat_prob} | {row.ml_confidence}")
        print(f"  Anomaly    : {row.anomaly_score}")
        print(f"  Surprise   : {row.yield_surprise} bpa")
        for k, v in row.items():
            if k.startswith("component_"):
                label = k.replace("component_", "").replace("_", " ").title()
                print(f"    {label:<14}: {v}")
    print("\n" + "=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live_csv",     default="quantagri_live/quantagri_live_results.csv")
    parser.add_argument("--analysis_csv", default="quantagri_analysis/quantagri_monthly_summary.csv")
    parser.add_argument("--model_dir",    default="ml_models")
    parser.add_argument("--output_csv",   default="quantagri_live/quantagri_signal_scorecard.csv")
    args = parser.parse_args()

    df = compute_signal_score(args.live_csv, args.analysis_csv, args.model_dir)
    df.to_csv(args.output_csv, index=False)
    print_scorecard(df)
    print(f"\nSaved to {args.output_csv}")
