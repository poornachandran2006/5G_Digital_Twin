"""
ml/shap_explainer.py

SHAP-based model explainability for the XGBoost congestion predictor.

Uses ``shap.TreeExplainer`` for efficient computation.  All plots are
saved to disk — no ``plt.show()`` calls.

Run from project root: python -m ml.shap_explainer
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required

import matplotlib.pyplot as plt
import numpy as np
import shap

from ml.xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainability wrapper for XGBoost congestion predictions.

    Uses :class:`shap.TreeExplainer` which is exact and fast for tree
    models.  All visualisations are saved to files — no interactive
    rendering.
    """

    def __init__(
        self,
        xgb_model: XGBoostPredictor,
        feature_names: list[str],
    ) -> None:
        """
        Args:
            xgb_model: Trained :class:`XGBoostPredictor` instance.
            feature_names: Ordered list of 18 feature names (must match
                the training column order).
        """
        self._model = xgb_model.model
        self.feature_names = feature_names
        self._explainer = shap.TreeExplainer(self._model)

    # ------------------------------------------------------------------
    # SHAP value computation
    # ------------------------------------------------------------------

    def compute_shap_values(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for each sample in ``X_flat``.

        Args:
            X_flat: Feature matrix ``(N, 18)``.

        Returns:
            np.ndarray: SHAP values array ``(N, 18)``.
        """
        sv = self._explainer.shap_values(X_flat)
        logger.info("SHAP values computed: shape %s", np.array(sv).shape)
        return np.array(sv)

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_summary(
        self,
        X_flat: np.ndarray,
        save_path: str = "reports/shap_summary.png",
    ) -> None:
        """
        Generate and save a beeswarm summary plot (top 10 features).

        Args:
            X_flat: Feature matrix ``(N, 18)``.
            save_path: Destination file path.
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        shap_values = self.compute_shap_values(X_flat)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_flat,
            feature_names=self.feature_names,
            max_display=10,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved to %s", save_path)

    def plot_waterfall(
        self,
        X_flat: np.ndarray,
        sample_idx: int = 0,
        save_path: str = "reports/shap_waterfall.png",
    ) -> None:
        """
        Generate and save a waterfall plot for a single prediction.

        Args:
            X_flat: Feature matrix ``(N, 18)``.
            sample_idx: Index of the sample to explain.
            save_path: Destination file path.
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        shap_values = self.compute_shap_values(X_flat)

        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=self._explainer.expected_value,
            feature_names=self.feature_names,
            data=X_flat[sample_idx],
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP waterfall plot saved to %s", save_path)

    # ------------------------------------------------------------------
    # Top features
    # ------------------------------------------------------------------

    def get_top_features(
        self,
        X_flat: np.ndarray,
        top_n: int = 5,
    ) -> list[Tuple[str, float]]:
        """
        Return the top-N features ranked by mean absolute SHAP value.

        Args:
            X_flat: Feature matrix ``(N, 18)``.
            top_n: Number of features to return.

        Returns:
            List of ``(feature_name, mean_abs_shap)`` tuples, sorted
            descending by importance.
        """
        shap_values = self.compute_shap_values(X_flat)
        mean_abs = np.abs(shap_values).mean(axis=0)

        ranked = sorted(
            zip(self.feature_names, mean_abs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        top = ranked[:top_n]
        logger.info("Top %d SHAP features: %s", top_n, top)
        return top
