"""
CLI trainer — trains and saves all models from historical data.

Usage:
    python quantagri/ml/train.py \
        --sat_csv quantagri_live/quantagri_live_results.csv \
        --yield_csv official_yields.csv \
        --model_dir ml_models
"""
import argparse
import pandas as pd
from pathlib import Path

from .features import build_season_features, add_lagged_features
from .models import YieldModel
from .anomaly import AnomalyDetector
from .phenology import PhenologyExtractor
from .signals import YieldSurpriseClassifier

COMBOS = [
    ("corn", "illinois_us"),
    ("corn", "iowa_us"),
    ("soy", "illinois_us"),
    ("soy", "mato_grosso_br"),
    ("wheat", "kansas_us"),
    ("wheat", "rostov_ru"),
]


def train_all(sat_csv: str, yield_csv: str, model_dir: str):
    sat = pd.read_csv(sat_csv)
    yld = pd.read_csv(yield_csv)
    yld = yld[yld.official_yield > 0]

    Path(model_dir).mkdir(exist_ok=True)

    pheno = PhenologyExtractor()
    pheno_df = pheno.batch_extract(sat)

    for commodity, region_id in COMBOS:
        print(f"\n── {commodity}/{region_id} ──")
        yld_sub = yld[(yld.commodity == commodity) & (yld.region_id == region_id)]

        # Features
        feats = build_season_features(sat, commodity, region_id)
        feats = add_lagged_features(feats)

        # Merge phenology
        pheno_sub = pheno_df[
            (pheno_df.commodity == commodity) & (pheno_df.region_id == region_id)
        ].drop(columns=["commodity", "region_id"], errors="ignore")
        feats = feats.merge(pheno_sub, on="season_year", how="left")

        if feats.empty or len(feats) < 3:
            print(f"  Skipping — insufficient data")
            continue

        # Yield model
        try:
            model = YieldModel(commodity, region_id, model_dir)
            cv = model.fit(feats, yld_sub)
            model.save()
            print(f"  YieldModel CV: {cv}")
        except Exception as e:
            print(f"  YieldModel failed: {e}")

        # Anomaly detector
        try:
            feat_cols = [c for c in feats.columns
                         if c not in ("season_year", "commodity", "region_id", "year")]
            detector = AnomalyDetector()
            detector.fit(feats, feat_cols)
            print(f"  AnomalyDetector fitted on {len(feats)} seasons")
        except Exception as e:
            print(f"  AnomalyDetector failed: {e}")

    print("\nAll models trained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sat_csv", required=True)
    parser.add_argument("--yield_csv", required=True)
    parser.add_argument("--model_dir", default="ml_models")
    args = parser.parse_args()
    train_all(args.sat_csv, args.yield_csv, args.model_dir)
