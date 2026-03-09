#!/usr/bin/env python3
"""
quantagri/ml/train.py

CLI trainer — trains and saves all models from historical + live data.
Designed to be re-run daily; skips retraining if insufficient new data.

Usage:
    python quantagri/ml/train.py \
        --sat_csv quantagri_live/quantagri_live_results.csv \
        --yield_csv official_yields.csv \
        --model_dir ml_models

    # Also include historical monthly data if available:
    python quantagri/ml/train.py \
        --sat_csv quantagri_live/quantagri_live_results.csv \
        --historical_csv quantagri_historical/quantagri_monthly_ALL.csv \
        --yield_csv official_yields.csv \
        --model_dir ml_models
"""
import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_season_features, add_lagged_features
from models import YieldModel
from anomaly import AnomalyDetector
from phenology import PhenologyExtractor
from signals import YieldSurpriseClassifier


# ── Constants ─────────────────────────────────────────────────────────────────

COMBOS = [
    ("corn",   "illinois_us"),
    ("corn",   "iowa_us"),
    ("soy",    "illinois_us"),
    ("soy",    "mato_grosso_br"),
    ("wheat",  "kansas_us"),
    ("wheat",  "rostov_ru"),
    ("cotton", "north_texas_us"),
    ("cotton", "south_texas_us"),
    ("cotton", "xinjiang_north_cn"),
    ("cotton", "xinjiang_tarim_cn"),
]

MIN_SEASONS_YIELD_MODEL      = 3   # minimum seasons to train YieldModel
MIN_SEASONS_CLASSIFIER       = 4   # minimum seasons to train classifier
MIN_SEASONS_ANOMALY          = 4   # minimum seasons to train anomaly detector
RETRAIN_THRESHOLD_NEW_ROWS   = 5   # only retrain if this many new rows since last train


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sat_data(sat_csv: str, historical_csv: str | None) -> pd.DataFrame:
    """
    Load and merge live results with historical monthly data if provided.
    Deduplicates by commodity + region_id + season_year + month.
    """
    frames = []

    if sat_csv and Path(sat_csv).exists():
        live = pd.read_csv(sat_csv)
        # Normalise column names — live monitor uses different names than historical
        live = _normalise_live_columns(live)
        frames.append(live)
        print(f"  Loaded live CSV: {len(live)} rows from {sat_csv}")
    else:
        print(f"  Warning: live CSV not found at {sat_csv}")

    if historical_csv and Path(historical_csv).exists():
        hist = pd.read_csv(historical_csv)
        hist = _normalise_historical_columns(hist)
        frames.append(hist)
        print(f"  Loaded historical CSV: {len(hist)} rows from {historical_csv}")
    else:
        if historical_csv:
            print(f"  Warning: historical CSV not found at {historical_csv}")
        print("  Tip: run quantagri_historical.yml first for best ML results")

    if not frames:
        raise FileNotFoundError("No satellite data found — cannot train")

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate — keep historical row if duplicate exists (more stable)
    combined = combined.drop_duplicates(
        subset=["commodity", "region_id", "season_year", "month"],
        keep="last"
    )
    print(f"  Combined satellite rows after dedup: {len(combined)}")
    return combined


def _normalise_live_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map live monitor column names to standard training schema."""
    rename = {
        "current_ndvi":          "ndvi_mean",
        "current_ndvi_velocity": "velocity_mean",
        "peak_ndvi":             "ndvi_max",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Derive season_year and month from as_of_date if not present
    if "season_year" not in df.columns and "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
        df["month"]       = df["as_of_date"].dt.month
        df["year"]        = df["as_of_date"].dt.year
        df["season_year"] = df["year"]  # approximate; historical runner sets this correctly

    return df


def _normalise_historical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Historical monthly CSVs already use the standard schema — minimal changes."""
    if "ndvi_mean" not in df.columns and "ndvi_mean_avg" in df.columns:
        df = df.rename(columns={"ndvi_mean_avg": "ndvi_mean"})
    return df


def load_yield_data(yield_csv: str) -> pd.DataFrame:
    """Load official yields, drop placeholder zeros."""
    yld = pd.read_csv(yield_csv)
    yld = yld[yld["official_yield"] > 0].copy()
    print(f"  Loaded yield CSV: {len(yld)} rows (after dropping zeros)")
    return yld


def should_retrain(model_dir: str, commodity: str, region_id: str,
                   sat_df: pd.DataFrame) -> bool:
    """
    Returns True if we should retrain this combo.
    Skips retrain if model is fresh and no significant new data has arrived.
    """
    meta_path = Path(model_dir) / f"{commodity}_{region_id}_meta.json"
    if not meta_path.exists():
        return True  # no model yet — always train

    with open(meta_path) as f:
        meta = json.load(f)

    last_row_count = meta.get("training_rows", 0)
    current_rows = len(sat_df[
        (sat_df.commodity == commodity) & (sat_df.region_id == region_id)
    ])

    if current_rows - last_row_count >= RETRAIN_THRESHOLD_NEW_ROWS:
        return True

    # Also retrain if last train was more than 7 days ago
    last_trained = meta.get("trained_at", "")
    if last_trained:
        try:
            last_dt = datetime.fromisoformat(last_trained)
            days_since = (datetime.now(timezone.utc) - last_dt).days
            if days_since >= 7:
                return True
        except ValueError:
            return True

    print(f"    Skipping retrain — only {current_rows - last_row_count} new rows "
          f"since last train ({last_row_count} rows)")
    return False


def save_meta(model_dir: str, commodity: str, region_id: str,
              n_rows: int, metrics: dict):
    """Save training metadata for retrain-gating logic."""
    meta = {
        "commodity":     commodity,
        "region_id":     region_id,
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "training_rows": n_rows,
        "metrics":       metrics,
    }
    path = Path(model_dir) / f"{commodity}_{region_id}_meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def build_consensus_df(yld: pd.DataFrame) -> pd.DataFrame:
    """
    Build a consensus yield DataFrame for the classifier.
    Uses a 3-year rolling mean of official yields as the 'consensus' proxy
    when no external consensus data is available.
    """
    rows = []
    for (commodity, region), grp in yld.groupby(["commodity", "region_id"]):
        grp = grp.sort_values("year").copy()
        grp["consensus_yield"] = grp["official_yield"].shift(1).rolling(3, min_periods=1).mean()
        grp = grp.dropna(subset=["consensus_yield"])
        rows.append(grp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ── Main trainer ──────────────────────────────────────────────────────────────

def train_all(
    sat_csv: str,
    yield_csv: str,
    model_dir: str,
    historical_csv: str | None = None,
    force: bool = False,
):
    Path(model_dir).mkdir(exist_ok=True)

    print("\n── Loading data ──────────────────────────────────────────────")
    sat = load_sat_data(sat_csv, historical_csv)
    yld = load_yield_data(yield_csv)
    consensus = build_consensus_df(yld)

    print("\n── Extracting phenology ──────────────────────────────────────")
    pheno = PhenologyExtractor()
    try:
        pheno_df = pheno.batch_extract(sat)
        print(f"  Phenology extracted for {len(pheno_df)} season/region combos")
    except Exception as e:
        pheno_df = pd.DataFrame()
        print(f"  Phenology extraction failed: {e}")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_rows = []

    for commodity, region_id in COMBOS:
        print(f"\n{'─' * 60}")
        print(f"  {commodity.upper()} / {region_id}")
        print(f"{'─' * 60}")

        sat_sub = sat[(sat.commodity == commodity) & (sat.region_id == region_id)]
        yld_sub = yld[(yld.commodity == commodity) & (yld.region_id == region_id)]

        n_sat_rows    = len(sat_sub)
        n_seasons_sat = sat_sub["season_year"].nunique() if "season_year" in sat_sub.columns else 0
        n_yield_years = len(yld_sub)

        print(f"  Satellite rows   : {n_sat_rows} ({n_seasons_sat} seasons)")
        print(f"  Yield years      : {n_yield_years}")

        if n_sat_rows == 0:
            print("  ⚠️  No satellite data — skipping all models")
            summary_rows.append(_summary_row(commodity, region_id, "skipped", "no satellite data"))
            continue

        # Check retrain gate
        if not force and not should_retrain(model_dir, commodity, region_id, sat):
            summary_rows.append(_summary_row(commodity, region_id, "skipped", "no new data"))
            continue

        # ── Build features ────────────────────────────────────────────────────
        try:
            feats = build_season_features(sat, commodity, region_id)
            feats = add_lagged_features(feats)

            # Merge phenology
            if not pheno_df.empty:
                pheno_sub = pheno_df[
                    (pheno_df.commodity == commodity) &
                    (pheno_df.region_id == region_id)
                ].drop(columns=["commodity", "region_id"], errors="ignore")
                feats = feats.merge(pheno_sub, on="season_year", how="left")

            print(f"  Feature seasons  : {len(feats)}")
        except Exception as e:
            print(f"  ❌ Feature engineering failed: {e}")
            summary_rows.append(_summary_row(commodity, region_id, "failed", f"features: {e}"))
            continue

        combo_metrics = {}

        # ── YieldModel ────────────────────────────────────────────────────────
        if len(feats) >= MIN_SEASONS_YIELD_MODEL and n_yield_years >= MIN_SEASONS_YIELD_MODEL:
            try:
                model = YieldModel(commodity, region_id, model_dir)
                cv = model.fit(feats, yld_sub)
                model.save()
                combo_metrics["yield_model"] = cv
                r2_str = f"Ridge R²={cv.get('ridge_cv_r2', 'n/a'):.3f}"
                if "lgbm_cv_r2" in cv:
                    r2_str += f" | LGBM R²={cv['lgbm_cv_r2']:.3f}"
                print(f"  ✅ YieldModel     : {r2_str}")
            except Exception as e:
                print(f"  ❌ YieldModel failed: {e}")
                combo_metrics["yield_model"] = {"error": str(e)}
        else:
            print(f"  ⚠️  YieldModel     : skipped "
                  f"(need {MIN_SEASONS_YIELD_MODEL} seasons, have {len(feats)})")

        # ── AnomalyDetector ───────────────────────────────────────────────────
        if len(feats) >= MIN_SEASONS_ANOMALY:
            try:
                feat_cols = [
                    c for c in feats.columns
                    if c not in ("season_year", "commodity", "region_id", "year")
                    and feats[c].notna().sum() >= MIN_SEASONS_ANOMALY
                ]
                detector = AnomalyDetector()
                detector.fit(feats, feat_cols)

                # Persist anomaly detector
                det_path = Path(model_dir) / f"{commodity}_{region_id}_anomaly.pkl"
                with open(det_path, "wb") as f:
                    pickle.dump(detector, f)

                combo_metrics["anomaly"] = {"fitted": True, "n_features": len(feat_cols)}
                print(f"  ✅ AnomalyDetector: fitted on {len(feats)} seasons, "
                      f"{len(feat_cols)} features")
            except Exception as e:
                print(f"  ❌ AnomalyDetector failed: {e}")
                combo_metrics["anomaly"] = {"error": str(e)}
        else:
            print(f"  ⚠️  AnomalyDetector: skipped "
                  f"(need {MIN_SEASONS_ANOMALY} seasons, have {len(feats)})")

        # ── YieldSurpriseClassifier ───────────────────────────────────────────
        if (len(feats) >= MIN_SEASONS_CLASSIFIER
                and n_yield_years >= MIN_SEASONS_CLASSIFIER
                and not consensus.empty):
            try:
                cons_sub = consensus[
                    (consensus.commodity == commodity) &
                    (consensus.region_id == region_id)
                ]
                if len(cons_sub) >= MIN_SEASONS_CLASSIFIER:
                    clf = YieldSurpriseClassifier()
                    # Pass yield_df with commodity/region_id for merge
                    yld_sub_full = yld_sub.copy()
                    yld_sub_full["commodity"] = commodity
                    yld_sub_full["region_id"] = region_id

                    feats_full = feats.copy()
                    feats_full["commodity"] = commodity
                    feats_full["region_id"] = region_id

                    report = clf.fit(feats_full, yld_sub_full, cons_sub)

                    clf_path = Path(model_dir) / f"{commodity}_{region_id}_classifier.pkl"
                    with open(clf_path, "wb") as f:
                        pickle.dump(clf, f)

                    combo_metrics["classifier"] = {
                        "feature_importance": report.get("feature_importance", {})
                    }
                    print(f"  ✅ Classifier     : fitted")
                    # Print top 3 features
                    fi = report.get("feature_importance", {})
                    if fi:
                        top3 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
                        print(f"     Top features  : "
                              f"{', '.join(f'{k}={v:.3f}' for k, v in top3)}")
                else:
                    print(f"  ⚠️  Classifier     : skipped "
                          f"(need {MIN_SEASONS_CLASSIFIER} consensus seasons, "
                          f"have {len(cons_sub)})")
            except Exception as e:
                print(f"  ❌ Classifier failed: {e}")
                combo_metrics["classifier"] = {"error": str(e)}
        else:
            print(f"  ⚠️  Classifier     : skipped "
                  f"(need {MIN_SEASONS_CLASSIFIER} seasons)")

        # ── Save metadata ─────────────────────────────────────────────────────
        save_meta(model_dir, commodity, region_id, n_sat_rows, combo_metrics)
        summary_rows.append(_summary_row(commodity, region_id, "trained", "", combo_metrics))

    # ── Print final summary ───────────────────────────────────────────────────
    _print_summary(summary_rows)

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary_path = Path(model_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "combos":     summary_rows,
        }, f, indent=2)
    print(f"\nTraining summary saved to {summary_path}")


# ── Utility ───────────────────────────────────────────────────────────────────

def _summary_row(commodity, region_id, status, note="", metrics=None):
    return {
        "commodity": commodity,
        "region_id": region_id,
        "status":    status,
        "note":      note,
        "metrics":   metrics or {},
    }


def _print_summary(rows: list):
    print(f"\n{'=' * 60}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 60}")
    for r in rows:
        icon = "✅" if r["status"] == "trained" else "⚠️ " if r["status"] == "skipped" else "❌"
        note = f" — {r['note']}" if r["note"] else ""
        print(f"  {icon} {r['commodity'].upper():<8} / {r['region_id']:<22} {r['status']}{note}")

        m = r.get("metrics", {})
        if "yield_model" in m and "ridge_cv_r2" in m["yield_model"]:
            cv = m["yield_model"]
            r2 = f"Ridge={cv['ridge_cv_r2']:.3f}"
            if "lgbm_cv_r2" in cv:
                r2 += f" LGBM={cv['lgbm_cv_r2']:.3f}"
            print(f"       YieldModel CV R²: {r2}")
    print(f"{'=' * 60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QuantAgri ML models")
    parser.add_argument(
        "--sat_csv",
        required=True,
        help="Path to live results CSV (quantagri_live/quantagri_live_results.csv)"
    )
    parser.add_argument(
        "--historical_csv",
        default=None,
        help="Path to historical monthly ALL CSV (quantagri_historical/quantagri_monthly_ALL.csv)"
    )
    parser.add_argument(
        "--yield_csv",
        required=True,
        help="Path to official yields CSV (official_yields.csv)"
    )
    parser.add_argument(
        "--model_dir",
        default="ml_models",
        help="Directory to save trained models (default: ml_models)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain even if no new data since last run"
    )
    args = parser.parse_args()

    train_all(
        sat_csv=args.sat_csv,
        yield_csv=args.yield_csv,
        model_dir=args.model_dir,
        historical_csv=args.historical_csv,
        force=args.force,
    )
