"""
quantagri/ml/models.py
LightGBM + Ridge ensemble yield model.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from features import FEAT_COLS


class YieldModel:
    """
    Ensemble of Ridge (40%) + LightGBM (60%) for yield prediction.
    Falls back to Ridge-only if LightGBM not available.
    """

    RIDGE_WEIGHT = 0.4
    LGBM_WEIGHT  = 0.6

    def __init__(self, commodity: str, region_id: str, model_dir: str = "ml_models"):
        self.commodity  = commodity
        self.region_id  = region_id
        self.model_dir  = Path(model_dir)
        self.scaler     = StandardScaler()
        self.ridge      = Ridge(alpha=1.0)
        self.lgbm       = None
        self.feat_cols_ = []
        self.fitted_    = False

    def _get_feat_cols(self, feats: pd.DataFrame) -> list:
        available = [c for c in FEAT_COLS if c in feats.columns]
        lag_cols  = [c for c in feats.columns if "_lag" in c]
        return available + lag_cols

    def fit(self, feats: pd.DataFrame, yield_df: pd.DataFrame) -> dict:
        """
        Fit the ensemble. Returns cross-validation metrics.
        feats:     season_year + feature columns
        yield_df:  year + official_yield columns
        """
        merged = feats.merge(
            yield_df[["year", "official_yield"]].rename(columns={"year": "season_year"}),
            on="season_year", how="inner"
        ).dropna()

        if len(merged) < 3:
            raise ValueError(f"Need at least 3 matched seasons, got {len(merged)}")

        self.feat_cols_ = self._get_feat_cols(merged)
        X = merged[self.feat_cols_].values
        y = merged["official_yield"].values

        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=min(3, len(merged) - 1))
        ridge_preds, lgbm_preds, actuals = [], [], []

        for train_idx, val_idx in tscv.split(X_scaled):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            r = Ridge(alpha=1.0).fit(X_tr, y_tr)
            ridge_preds.extend(r.predict(X_val))
            actuals.extend(y_val)

            if HAS_LGBM and len(X_tr) >= 3:
                lg = lgb.LGBMRegressor(
                    n_estimators=100, learning_rate=0.05,
                    num_leaves=15, min_child_samples=2, verbose=-1
                ).fit(X_tr, y_tr)
                lgbm_preds.extend(lg.predict(X_val))

        cv = {}
        if actuals:
            cv["ridge_cv_r2"] = float(r2_score(actuals, ridge_preds))
            if lgbm_preds:
                cv["lgbm_cv_r2"] = float(r2_score(actuals, lgbm_preds))

        # Final fit on all data
        self.ridge.fit(X_scaled, y)

        if HAS_LGBM and len(X_scaled) >= 3:
            self.lgbm = lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.05,
                num_leaves=15, min_child_samples=2, verbose=-1
            ).fit(X_scaled, y)

        self.fitted_ = True
        return cv

    def predict(self, feat_row: pd.DataFrame) -> dict:
        if not self.fitted_:
            raise RuntimeError("Model not fitted")

        X = feat_row[self.feat_cols_].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        ridge_pred = float(self.ridge.predict(X_scaled)[0])

        if self.lgbm is not None:
            lgbm_pred    = float(self.lgbm.predict(X_scaled)[0])
            ensemble_pred = self.RIDGE_WEIGHT * ridge_pred + self.LGBM_WEIGHT * lgbm_pred
        else:
            lgbm_pred     = None
            ensemble_pred = ridge_pred

        return {
            "ridge_pred":    ridge_pred,
            "lgbm_pred":     lgbm_pred,
            "ensemble_pred": ensemble_pred,
        }

    def save(self):
        self.model_dir.mkdir(exist_ok=True)
        path = self.model_dir / f"{self.commodity}_{self.region_id}_yield.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, commodity: str, region_id: str, model_dir: str = "ml_models"):
        path = Path(model_dir) / f"{commodity}_{region_id}_yield.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
