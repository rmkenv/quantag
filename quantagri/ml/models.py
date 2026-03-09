"""
Yield prediction models.
Ensemble of LightGBM + Ridge with time-series cross-validation.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from .features import FEAT_COLS


class YieldModel:
    """
    Ensemble yield regressor.
    Uses LightGBM if available, falls back to Ridge.
    Trained per commodity+region.
    """

    def __init__(self, commodity: str, region_id: str, model_dir: str = "ml_models"):
        self.commodity = commodity
        self.region_id = region_id
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.scaler = StandardScaler()
        self.ridge = Ridge(alpha=1.0)
        self.lgbm = LGBMRegressor(
            n_estimators=100, learning_rate=0.05,
            max_depth=3, min_child_samples=2,
            verbose=-1,
        ) if HAS_LGBM else None
        self.fitted = False
        self.feature_cols = None
        self.cv_scores = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, feat_df: pd.DataFrame, yield_df: pd.DataFrame) -> dict:
        """
        feat_df: output of build_season_features()
        yield_df: columns [year, official_yield]
        Returns dict of CV metrics.
        """
        merged = feat_df.merge(
            yield_df[["year", "official_yield"]],
            left_on="season_year", right_on="year", how="inner"
        )
        available_cols = [c for c in FEAT_COLS if c in merged.columns]
        merged = merged.dropna(subset=available_cols)

        if len(merged) < 3:
            raise ValueError(f"Insufficient data: {len(merged)} rows after merge/dropna")

        self.feature_cols = available_cols
        X = merged[available_cols].values
        y = merged["official_yield"].values

        X_s = self.scaler.fit_transform(X)
        self.ridge.fit(X_s, y)

        if self.lgbm and len(merged) >= 5:
            self.lgbm.fit(X, y)

        # Time-series CV
        self.cv_scores = self._time_series_cv(X, X_s, y)
        self.fitted = True
        return self.cv_scores

    def _time_series_cv(self, X_raw, X_scaled, y) -> dict:
        n_splits = min(3, len(y) - 2)
        if n_splits < 2:
            return {}
        tscv = TimeSeriesSplit(n_splits=n_splits)
        ridge_r2, lgbm_r2 = [], []

        for train_idx, test_idx in tscv.split(X_scaled):
            # Ridge
            s = StandardScaler().fit(X_scaled[train_idx])
            r = Ridge(alpha=1.0).fit(s.transform(X_scaled[train_idx]), y[train_idx])
            ridge_r2.append(r2_score(y[test_idx], r.predict(s.transform(X_scaled[test_idx]))))

            # LightGBM
            if self.lgbm and len(train_idx) >= 3:
                lg = LGBMRegressor(n_estimators=100, learning_rate=0.05,
                                   max_depth=3, min_child_samples=2, verbose=-1)
                lg.fit(X_raw[train_idx], y[train_idx])
                lgbm_r2.append(r2_score(y[test_idx], lg.predict(X_raw[test_idx])))

        scores = {"ridge_cv_r2": float(np.mean(ridge_r2))}
        if lgbm_r2:
            scores["lgbm_cv_r2"] = float(np.mean(lgbm_r2))
        return scores

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, feat_row: pd.Series) -> dict:
        """
        Predict yield for a single season feature row.
        Returns dict with ridge_pred, lgbm_pred, ensemble_pred.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted — call fit() first")

        x = feat_row[self.feature_cols].values.reshape(1, -1)
        x_s = self.scaler.transform(x)

        ridge_pred = float(self.ridge.predict(x_s)[0])
        result = {"ridge_pred": ridge_pred}

        if self.lgbm and self.lgbm.n_estimators_:
            lgbm_pred = float(self.lgbm.predict(x)[0])
            result["lgbm_pred"] = lgbm_pred
            result["ensemble_pred"] = 0.4 * ridge_pred + 0.6 * lgbm_pred
        else:
            result["ensemble_pred"] = ridge_pred

        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        path = self.model_dir / f"{self.commodity}_{self.region_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, commodity: str, region_id: str, model_dir: str = "ml_models"):
        path = Path(model_dir) / f"{commodity}_{region_id}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
