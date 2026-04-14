"""
Anomaly Detection for 5G KPI Stream
Uses Isolation Forest (sklearn) — unsupervised, no labels needed.
Trained once on the full KPI dataset, then scores every live tick.
"""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("anomaly_detector")

# These must match FEATURE_COLUMNS in data_preprocessor.py (same 18 features)
FEATURE_COLUMNS = [
    "cell0_load", "cell1_load", "cell2_load",
    "cell0_throughput", "cell1_throughput", "cell2_throughput",
    "cell0_ue_count", "cell1_ue_count", "cell2_ue_count",
    "cell0_avg_sinr", "cell1_avg_sinr", "cell2_avg_sinr",
    "system_throughput", "system_avg_sinr", "system_avg_latency_ms",
    "handover_count", "handover_rate", "packet_loss_rate",
]


class AnomalyDetector:
    """
    Wraps sklearn IsolationForest for live KPI anomaly scoring.

    contamination=0.05 means we expect ~5% of ticks to be anomalous.
    This is a reasonable assumption for a 5G network with injected congestion.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=100,       # 100 trees — good balance of speed vs accuracy
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,              # use all CPU cores
        )
        self.is_fitted = False
        self.contamination = contamination

    def fit(self, X: np.ndarray) -> None:
        """
        Train the Isolation Forest on historical KPI data.
        X shape: (n_samples, 18)
        """
        self.model.fit(X)
        self.is_fitted = True
        logger.info("AnomalyDetector fitted on %d samples", X.shape[0])

    def score(self, feature_row: np.ndarray) -> dict:
        """
        Score a single tick.
        feature_row shape: (18,) — the same 18 features used in training.

        Returns a dict with:
          - anomaly_score: float 0.0 to 1.0 (higher = more anomalous)
          - is_anomaly: bool (True if score > threshold)
          - severity: "normal" | "warning" | "critical"
        """
        if not self.is_fitted:
            return {"anomaly_score": 0.0, "is_anomaly": False, "severity": "normal"}

        x = feature_row.reshape(1, -1)

        # decision_function returns negative scores for anomalies
        # We flip and normalise to [0, 1] range
        raw_score = float(self.model.decision_function(x)[0])

        # sklearn range is roughly [-0.5, 0.5] — normalise to [0, 1]
        # anomaly_score close to 1.0 = very anomalous
        anomaly_score = float(np.clip(0.5 - raw_score, 0.0, 1.0))

        is_anomaly = bool(self.model.predict(x)[0] == -1)  # -1 = anomaly in sklearn

        # Severity thresholds
        if anomaly_score > 0.75:
            severity = "critical"
        elif anomaly_score > 0.55:
            severity = "warning"
        else:
            severity = "normal"

        return {
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": is_anomaly,
            "severity": severity,
        }

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("AnomalyDetector saved to %s", path)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        logger.info("AnomalyDetector loaded from %s", path)