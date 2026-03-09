"""
Yield surprise classifier.
Predicts long/short signal: will actual yield beat or miss consensus?
Includes SHAP explanation support.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

from .features import FEAT_COLS


class YieldSurpriseClassifier:
    """
    Binary classifier: 1 = yield will beat USDA/CONAB consensus,
                       0 = yield will miss.
    """

    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            class_weight="balanced",
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_cols = None
        self.threshold = 0.5

    def fit(self, feat_df: pd.DataFrame, yield_df: pd.DataFrame,
            consensus_df: pd.DataFrame) -> dict:
        """
        consensus_df: columns [year, commodity, region_id, consensus_yield]
        Label = 1 if official_yield > consensus_yield.
        """
        merged = feat_df.merge(
            yield_df[["year", "commodity", "region_id", "official_yield"]],
            left_on=["season_year", "commodity", "region_id"],
            right_on=["year", "commodity", "region_id"], how="inner"
        ).merge(
            consensus_df[["year", "commodity", "region_id", "consensus_yield"]],
            on=["year", "commodity", "region_id"], how="inner"
        )

        merged["beat"] = (merged["official_yield"] > merged["consensus_yield"]).astype(int)
        available_cols = [c for c in FEAT_COLS if c in merged.columns]
        merged = merged.dropna(subset=available_cols)

        if len(merged) < 4:
            raise ValueError(f"Need at least 4 samples, got {len(merged)}")

        self.feature_cols = available_cols
        X = merged[available_cols].values
        y = merged["beat"].values

        X_s = self.scaler.fit_transform(X)
        self.clf.fit(X_s, y)
        self.fitted = True

        # CV report
        tscv = TimeSeriesSplit(n_splits=min(3, len(y) - 2))
        preds, trues = [], []
        for train_idx, test_idx in tscv.split(X_s):
            clf_cv = RandomForestClassifier(
                n_estimators=200, max_depth=4,
                class_weight="balanced", random_state=42
            )
            s = StandardScaler().fit(X[train_idx])
            clf_cv.fit(s.transform(X[train_idx]), y[train_idx])
            preds.extend(clf_cv.predict(s.transform(X[test_idx])))
            trues.extend(y[test_idx])

        return {
            "report": classification_report(trues, preds, zero_division=0),
            "feature_importance": dict(zip(available_cols, self.clf.feature_importances_)),
        }

    def predict_signal(self, feat_row: pd.Series) -> dict:
        """
        Returns signal dict:
        {
          "signal": "LONG" | "SHORT" | "NEUTRAL",
          "beat_probability": float,
          "confidence": "high" | "medium" | "low"
        }
        """
        if not self.fitted:
            return {"signal": "NEUTRAL", "beat_probability": 0.5, "confidence": "low"}

        x = feat_row[self.feature_cols].values.reshape(1, -1)
        x_s = self.scaler.transform(x)
        proba = float(self.clf.predict_proba(x_s)[0][1])

        if proba >= 0.65:
            signal, confidence = "LONG", "high" if proba >= 0.75 else "medium"
        elif proba <= 0.35:
            signal, confidence = "SHORT", "high" if proba <= 0.25 else "medium"
        else:
            signal, confidence = "NEUTRAL", "low"

        return {
            "signal": signal,
            "beat_probability": round(proba, 4),
            "confidence": confidence,
        }

    def explain(self, feat_row: pd.Series) -> dict:
        """SHAP values for a single prediction (requires shap package)."""
        try:
            import shap
            explainer = shap.TreeExplainer(self.clf)
            x = feat_row[self.feature_cols].values.reshape(1, -1)
            x_s = self.scaler.transform(x)
            shap_vals = explainer.shap_values(x_s)
            # shap_vals[1] = contribution toward "beat" class
            contributions = dict(zip(self.feature_cols, shap_vals[1][0]))
            return dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
        except ImportError:
            return {"error": "pip install shap to enable explanations"}
