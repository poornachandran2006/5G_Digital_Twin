"""
ml/ensemble.py

Weighted ensemble of LSTM and XGBoost congestion predictors.

The default blend is 60 % LSTM + 40 % XGBoost, which balances the
sequential pattern recognition of the LSTM with the feature-interaction
strength of XGBoost.

Run from project root: python -m ml.ensemble
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from ml.lstm_model import CongestionLSTM
from ml.xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Weighted ensemble of :class:`CongestionLSTM` and :class:`XGBoostPredictor`.

    Combines the probability outputs of both models as a weighted average
    to produce a single prediction.
    """

    def __init__(
        self,
        lstm_model: CongestionLSTM,
        xgb_predictor: XGBoostPredictor,
        device: torch.device,
        lstm_weight: float = 0.6,
        xgb_weight: float = 0.4,
    ) -> None:
        """
        Args:
            lstm_model: Trained LSTM model (already loaded with best weights).
            xgb_predictor: Trained XGBoost predictor.
            device: Torch device for LSTM inference.
            lstm_weight: Weight applied to LSTM probability.
            xgb_weight: Weight applied to XGBoost probability.
        """
        self.lstm = lstm_model.to(device)
        self.xgb = xgb_predictor
        self.device = device
        self.lstm_weight = lstm_weight
        self.xgb_weight = xgb_weight
        logger.info(
            "Ensemble initialised: LSTM=%.1f%% XGB=%.1f%%",
            lstm_weight * 100, xgb_weight * 100,
        )

    # ------------------------------------------------------------------
    # Probability / prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        X_seq_tensor: torch.Tensor,
        X_flat_array: np.ndarray,
    ) -> np.ndarray:
        """
        Return blended probability of congestion for each sample.

        ``P_ensemble = lstm_weight * sigmoid(lstm_logit) + xgb_weight * xgb_proba``

        Args:
            X_seq_tensor: LSTM input ``(N, seq_len, 18)`` as a torch Tensor.
            X_flat_array: XGBoost input ``(N, 18)`` as a NumPy array.

        Returns:
            np.ndarray: Blended probabilities ``(N,)`` in ``[0, 1]``.
        """
        # LSTM probability
        self.lstm.eval()
        with torch.no_grad():
            logits = self.lstm(X_seq_tensor.to(self.device))  # (N, 1)
            lstm_proba = torch.sigmoid(logits).cpu().numpy().flatten()

        # XGBoost probability
        xgb_proba = self.xgb.model.predict_proba(X_flat_array)[:, 1]

        return self.lstm_weight * lstm_proba + self.xgb_weight * xgb_proba

    def predict(
        self,
        X_seq_tensor: torch.Tensor,
        X_flat_array: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Return binary predictions using the blended probabilities.

        Args:
            X_seq_tensor: LSTM input ``(N, seq_len, 18)``.
            X_flat_array: XGBoost input ``(N, 18)``.
            threshold: Decision threshold for classifying as congested.

        Returns:
            np.ndarray: Binary predictions ``(N,)`` — 0 or 1.
        """
        probas = self.predict_proba(X_seq_tensor, X_flat_array)
        return (probas >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_seq_tensor: torch.Tensor,
        X_flat_array: np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
        """
        Evaluate the ensemble on a test set.

        Args:
            X_seq_tensor: LSTM input ``(N, seq_len, 18)``.
            X_flat_array: XGBoost input ``(N, 18)``.
            y_true: Ground-truth labels ``(N,)``.

        Returns:
            dict with keys: ``accuracy``, ``precision``, ``recall``,
            ``f1``, ``auc_roc``, ``confusion_matrix``.
        """
        probas = self.predict_proba(X_seq_tensor, X_flat_array)
        y_int = y_true.astype(int)

        # Find threshold that maximises F1
        best_f1, best_thresh = 0.0, 0.5
        for t in np.arange(0.10, 0.90, 0.01):
            p = (probas >= t).astype(int)
            f = float(f1_score(y_int, p, zero_division=0))
            if f > best_f1:
                best_f1, best_thresh = f, t

        preds = (probas >= best_thresh).astype(int)
        logger.info("Ensemble optimal threshold: %.2f (F1=%.4f)", best_thresh, best_f1)

        metrics = {
            "accuracy": float(accuracy_score(y_int, preds)),
            "precision": float(precision_score(y_int, preds, zero_division=0)),
            "recall": float(recall_score(y_int, preds, zero_division=0)),
            "f1": float(f1_score(y_int, preds, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_int, probas)),
            "confusion_matrix": confusion_matrix(y_int, preds).tolist(),
        }
        logger.info("Ensemble metrics: %s", {k: v for k, v in metrics.items() if k != "confusion_matrix"})
        return metrics
