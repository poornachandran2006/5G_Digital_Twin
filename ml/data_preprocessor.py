"""
ml/data_preprocessor.py

Data loading, sequencing, splitting, and scaling for the ML pipeline.

All operations preserve temporal order (no shuffle).  Scaler is fit on
train data only — no data leakage.

Run from project root: python -m ml.data_preprocessor
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)

# Feature columns (indices 2-19 in the Phase 3 CSV)
FEATURE_COLUMNS: list[str] = [
    "cell0_load", "cell1_load", "cell2_load",
    "cell0_throughput", "cell1_throughput", "cell2_throughput",
    "cell0_ue_count", "cell1_ue_count", "cell2_ue_count",
    "cell0_avg_sinr", "cell1_avg_sinr", "cell2_avg_sinr",
    "system_throughput", "system_avg_sinr", "system_avg_latency_ms",
    "handover_count", "handover_rate", "packet_loss_rate",
]

LABEL_COLUMN: str = "is_congested"


class DataPreprocessor:
    """
    End-to-end data preparation for the 5G congestion prediction pipeline.

    Responsibilities:
    - Load and validate CSV data
    - Create LSTM sequences with configurable horizon
    - Time-ordered stratified splits (train / val / test)
    - StandardScaler fit on train only
    - Class-weight computation for imbalanced binary classification
    """

    def __init__(self) -> None:
        """Initialise the preprocessor with config-driven defaults."""
        self.seq_len: int = config.SEQUENCE_LENGTH
        self.horizon: int = config.PREDICTION_HORIZON
        self.scaler: StandardScaler = StandardScaler()
        self.feature_names: list[str] = list(FEATURE_COLUMNS)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, csv_path: str = "data/kpi_dataset.csv") -> pd.DataFrame:
        """
        Load the KPI dataset CSV and validate its shape.

        Args:
            csv_path: Path to the CSV file produced by Phase 3.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            ValueError: If the dataset does not have the expected shape.
        """
        df = pd.read_csv(csv_path)
        logger.info("Loaded %s: shape %s", csv_path, df.shape)

        if df.shape[1] != 22:
            raise ValueError(
                f"Expected 22 columns, got {df.shape[1]}. "
                f"Columns: {list(df.columns)}"
            )

        # Validate feature and label columns exist
        missing = set(FEATURE_COLUMNS + [LABEL_COLUMN]) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        logger.info(
            "Label distribution: %s",
            df[LABEL_COLUMN].value_counts().to_dict(),
        )
        return df

    # ------------------------------------------------------------------
    # Sequence creation
    # ------------------------------------------------------------------

    def create_sequences(
        self,
        df: pd.DataFrame,
        seq_len: int | None = None,
        horizon: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build sliding-window sequences for LSTM and flat features for XGBoost.

        For each valid starting index *i*, a sequence is the feature window
        ``[i : i + seq_len]`` and the label is
        ``is_congested[i + seq_len - 1 + horizon]``  (predict *horizon* ticks
        into the future from the end of the window).

        Args:
            df: Full dataset DataFrame.
            seq_len: LSTM look-back window (default: ``SEQUENCE_LENGTH``).
            horizon: Prediction horizon in ticks (default: ``PREDICTION_HORIZON``).

        Returns:
            Tuple of:
            - **X_seq** ``(N, seq_len, 18)`` — LSTM input sequences.
            - **y** ``(N,)`` — binary labels.
            - **X_flat** ``(N, 18)`` — last tick of each window (for XGBoost).
        """
        if seq_len is None:
            seq_len = self.seq_len
        if horizon is None:
            horizon = self.horizon

        features = df[FEATURE_COLUMNS].values.astype(np.float32)
        labels = df[LABEL_COLUMN].values.astype(np.float32)

        n_samples = len(df) - seq_len - horizon + 1
        if n_samples <= 0:
            raise ValueError(
                f"Not enough data: {len(df)} rows for "
                f"seq_len={seq_len}, horizon={horizon}"
            )

        X_seq = np.empty((n_samples, seq_len, len(FEATURE_COLUMNS)), dtype=np.float32)
        y = np.empty(n_samples, dtype=np.float32)

        for i in range(n_samples):
            X_seq[i] = features[i : i + seq_len]
            y[i] = labels[i + seq_len - 1 + horizon]

        # X_flat: last tick of each window (for tree-based models)
        X_flat = X_seq[:, -1, :].copy()

        logger.info(
            "Sequences created: X_seq=%s, y=%s, X_flat=%s (pos_rate=%.2f%%)",
            X_seq.shape, y.shape, X_flat.shape,
            y.mean() * 100,
        )
        return X_seq, y, X_flat

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split_data(
        self,
        X_seq: np.ndarray,
        X_flat: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.15,
    ) -> dict[str, np.ndarray]:
        """
        Time-ordered split into train / validation / test sets.

        No shuffling — temporal order is preserved to prevent data leakage.
        Stratification is implicit because the congestion injection intervals
        are spread across the timeline.

        Args:
            X_seq: LSTM sequences ``(N, seq_len, 18)``.
            X_flat: Flat features ``(N, 18)``.
            y: Labels ``(N,)``.
            test_size: Fraction for the test set (from the end).
            val_size: Fraction for the validation set (before test).

        Returns:
            dict with keys:
            ``X_seq_train``, ``X_seq_val``, ``X_seq_test``,
            ``X_flat_train``, ``X_flat_val``, ``X_flat_test``,
            ``y_train``, ``y_val``, ``y_test``.
        """
        n = len(y)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        n_train = n - n_val - n_test

        splits = {
            "X_seq_train":  X_seq[:n_train],
            "X_seq_val":    X_seq[n_train : n_train + n_val],
            "X_seq_test":   X_seq[n_train + n_val :],
            "X_flat_train": X_flat[:n_train],
            "X_flat_val":   X_flat[n_train : n_train + n_val],
            "X_flat_test":  X_flat[n_train + n_val :],
            "y_train":      y[:n_train],
            "y_val":        y[n_train : n_train + n_val],
            "y_test":       y[n_train + n_val :],
        }

        logger.info(
            "Split sizes: train=%d, val=%d, test=%d (total=%d)",
            n_train, n_val, n_test, n,
        )
        for name in ("y_train", "y_val", "y_test"):
            arr = splits[name]
            logger.info(
                "  %s: pos_rate=%.2f%% (%d / %d)",
                name, arr.mean() * 100, int(arr.sum()), len(arr),
            )
        return splits

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def scale_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        scaler_path: str = "models/scaler.pkl",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit a :class:`StandardScaler` on **train only**, then transform
        val and test.

        The scaler is persisted to ``scaler_path`` for inference-time use.

        Args:
            X_train: Training flat features ``(N_train, 18)``.
            X_val: Validation flat features ``(N_val, 18)``.
            X_test: Test flat features ``(N_test, 18)``.
            scaler_path: Where to save the fitted scaler.

        Returns:
            Tuple of (X_train_scaled, X_val_scaled, X_test_scaled).
        """
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train).astype(np.float32)
        X_val_s = self.scaler.transform(X_val).astype(np.float32)
        X_test_s = self.scaler.transform(X_test).astype(np.float32)

        joblib.dump(self.scaler, scaler_path)
        logger.info("Scaler fitted on train (%d samples), saved to %s",
                     len(X_train), scaler_path)
        return X_train_s, X_val_s, X_test_s

    def scale_sequences(
        self,
        X_seq_train: np.ndarray,
        X_seq_val: np.ndarray,
        X_seq_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply the already-fitted scaler to 3-D LSTM sequences.

        The scaler must have been fitted via :meth:`scale_features` first.

        Args:
            X_seq_train: ``(N_train, seq_len, 18)``
            X_seq_val: ``(N_val, seq_len, 18)``
            X_seq_test: ``(N_test, seq_len, 18)``

        Returns:
            Tuple of scaled sequences with the same shapes.
        """
        def _scale_3d(arr: np.ndarray) -> np.ndarray:
            n, s, f = arr.shape
            return self.scaler.transform(
                arr.reshape(-1, f)
            ).reshape(n, s, f).astype(np.float32)

        return _scale_3d(X_seq_train), _scale_3d(X_seq_val), _scale_3d(X_seq_test)

    # ------------------------------------------------------------------
    # Class weight
    # ------------------------------------------------------------------

    @staticmethod
    def compute_class_weight(y_train: np.ndarray) -> float:
        """
        Compute ``pos_weight`` for ``BCEWithLogitsLoss`` to handle class
        imbalance.

        Formula: ``neg_count / pos_count``.  For 12.1 % positive rate this
        is approximately 7.3.

        Args:
            y_train: Binary training labels ``(N,)``.

        Returns:
            float: Weight to apply to the positive class.
        """
        pos = float(y_train.sum())
        neg = float(len(y_train) - pos)
        weight = neg / max(pos, 1.0)
        logger.info("Class weight: neg=%d, pos=%d -> pos_weight=%.2f", int(neg), int(pos), weight)
        return weight
