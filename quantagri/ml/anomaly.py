"""
Anomaly detection on intra-season NDVI/LSWI trajectories.
Flags when current season deviates from historical envelope.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Trained on historical season feature vectors.
    Scores new seasons — negative = anomalous.
    """

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.iso = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_cols = None

    def fit(self, feat_df: pd.DataFrame, feature_cols: list):
        df = feat_df[feature_cols].dropna()
        if len(df) < 4:
            return self
        self.feature_cols = feature_cols
        X = self.scaler.fit_transform(df.values)
        self.iso.fit(X)
        self.fitted = True
        return self

    def score(self, feat_row: pd.Series) -> float:
        """
        Returns anomaly score. More negative = more anomalous.
        Threshold: < -0.05 is worth flagging.
        """
        if not self.fitted:
            return 0.0
        x = feat_row[self.feature_cols].values.reshape(1, -1)
        x_s = self.scaler.transform(x)
        return float(self.iso.decision_function(x_s)[0])

    def is_anomaly(self, feat_row: pd.Series, threshold: float = -0.05) -> bool:
        return self.score(feat_row) < threshold

    def historical_envelope(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns mean ± 2 std envelope for each feature.
        Useful for dashboard plotting.
        """
        if not self.feature_cols:
            return pd.DataFrame()
        df = feat_df[self.feature_cols].dropna()
        return pd.DataFrame({
            "feature": self.feature_cols,
            "mean": df.mean().values,
            "std": df.std().values,
            "lower": (df.mean() - 2 * df.std()).values,
            "upper": (df.mean() + 2 * df.std()).values,
        })
