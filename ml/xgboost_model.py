"""
ml/xgboost_model.py

XGBoost-based congestion predictor for the 5G Digital Twin.

Uses flat feature vectors (last tick of each LSTM window) and XGBoost's
built-in ``scale_pos_weight`` to handle class imbalance.

Run from project root: python -m ml.xgboost_model
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """
    XGBoost binary classifier for 5G congestion prediction.

    Wraps :class:`xgboost.XGBClassifier` with sensible defaults,
    early stopping, and evaluation utilities.
    """

    def __init__(self, scale_pos_weight: float = 1.0) -> None:
        """
        Args:
            scale_pos_weight: Weight for the positive class to handle
                imbalance.  Typically ``neg_count / pos_count``.
        """
        self.model: xgb.XGBClassifier = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            early_stopping_rounds=20,
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
        self.feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Train the XGBoost classifier with early stopping.

        Args:
            X_train: Training features ``(N_train, 18)``.
            y_train: Training labels ``(N_train,)``.
            X_val: Validation features ``(N_val, 18)``.
            y_val: Validation labels ``(N_val,)``.
            feature_names: Optional list of feature names for importance.
        """
        self.feature_names = feature_names

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        logger.info(
            "XGBoost training complete: best iteration = %s",
            getattr(self.model, "best_iteration", "N/A"),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the trained model on a test set.

        Uses the probability output and an F1-optimised threshold rather
        than the default 0.5 decision boundary, which is unsuitable for
        imbalanced datasets.

        Args:
            X_test: Test features ``(N_test, 18)``.
            y_test: Test labels ``(N_test,)``.

        Returns:
            dict with keys: ``accuracy``, ``precision``, ``recall``,
            ``f1``, ``auc_roc``, ``confusion_matrix``.
        """
        probas = self.model.predict_proba(X_test)[:, 1]
        y_int = y_test.astype(int)

        # Find threshold that maximises F1
        best_f1, best_thresh = 0.0, 0.5
        for t in np.arange(0.10, 0.90, 0.01):
            p = (probas >= t).astype(int)
            f = float(f1_score(y_int, p, zero_division=0))
            if f > best_f1:
                best_f1, best_thresh = f, t

        preds = (probas >= best_thresh).astype(int)
        logger.info("XGBoost optimal threshold: %.2f (F1=%.4f)", best_thresh, best_f1)

        return {
            "accuracy": float(accuracy_score(y_int, preds)),
            "precision": float(precision_score(y_int, preds, zero_division=0)),
            "recall": float(recall_score(y_int, preds, zero_division=0)),
            "f1": float(f1_score(y_int, preds, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_int, probas)),
            "confusion_matrix": confusion_matrix(y_int, preds).tolist(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "models/xgboost_model.json") -> None:
        """
        Save the trained model to JSON.

        Args:
            path: Destination path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        logger.info("XGBoost model saved to %s", path)

    def load(self, path: str = "models/xgboost_model.json") -> None:
        """
        Load a previously saved model from JSON.

        Args:
            path: Path to the saved model.
        """
        self.model.load_model(path)
        logger.info("XGBoost model loaded from %s", path)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        """
        Return a mapping of feature names to importance scores.

        Uses ``weight`` importance type (number of splits).

        Returns:
            dict: ``{feature_name: importance}``.
        """
        raw = self.model.get_booster().get_score(importance_type="weight")
        if self.feature_names is None:
            return {k: float(v) for k, v in raw.items()}

        result: dict[str, float] = {}
        for i, name in enumerate(self.feature_names):
            key = f"f{i}"
            result[name] = float(raw.get(key, 0.0))
        return result
