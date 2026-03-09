#!/usr/bin/env python3
"""
quantagri_email_summary.py
Reads the live results CSV and prints a plain-text email summary to stdout.
Usage: python3 quantagri/quantagri_email_summary.py quantagri_live/quantagri_live_results.csv
"""
import sys
import pandas as pd
from datetime import datetime

def main():
    if len(sys.argv) < 2:
        print("Usage: quantagri_email_summary.py <results_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"No results file found at {csv_path} — pipeline may not have run yet.")
        sys.exit(0)

    if df.empty:
        print("No results to summarize.")
        sys.exit(0)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    latest = df.sort_values("run_date").groupby(["commodity", "region_id"]).tail(1)

    lines = [
        f"QuantAgri Daily Summary — {today}",
        "=" * 50,
        "",
    ]

    for _, row in latest.iterrows():
        lines.append(f"  {row.get('commodity','').upper()} / {row.get('region_id','')}")
        for col in ["ndvi_mean", "lswi_mean", "velocity_mean", "yield_surprise_bpa"]:
            if col in row and pd.notna(row[col]):
                lines.append(f"    {col}: {row[col]:.4f}")
        lines.append("")

    alert_col = "alert_flag" if "alert_flag" in df.columns else None
    if alert_col:
        alerts = latest[latest[alert_col] == True]
        if not alerts.empty:
            lines.append("⚠️  ALERTS TRIGGERED:")
            for _, row in alerts.iterrows():
                lines.append(f"  → {row.get('commodity','').upper()} / {row.get('region_id','')} flagged")
            lines.append("")

    print("\n".join(lines))

if __name__ == "__main__":
    main()
