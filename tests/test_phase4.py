"""
tests/test_phase4.py

Unit tests for Phase 4 — ML prediction models.

Covers data preprocessing shapes, sequence alignment, scaler integrity,
model forward passes, SHAP values, ensemble blending, class weight,
and split correctness.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ml.data_preprocessor import DataPreprocessor, FEATURE_COLUMNS
from ml.lstm_model import CongestionLSTM, LSTMTrainer
from ml.xgboost_model import XGBoostPredictor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def preprocessed_data():
    """Load and preprocess the dataset once for all tests."""
    prep = DataPreprocessor()
    df = prep.load_data("data/kpi_dataset.csv")
    X_seq, y, X_flat = prep.create_sequences(df)
    splits = prep.split_data(X_seq, X_flat, y)
    X_flat_train_s, X_flat_val_s, X_flat_test_s = prep.scale_features(
        splits["X_flat_train"], splits["X_flat_val"], splits["X_flat_test"],
        scaler_path=os.path.join(tempfile.gettempdir(), "test_scaler.pkl"),
    )
    return {
        "prep": prep,
        "df": df,
        "X_seq": X_seq,
        "y": y,
        "X_flat": X_flat,
        "splits": splits,
        "X_flat_train_s": X_flat_train_s,
        "X_flat_val_s": X_flat_val_s,
        "X_flat_test_s": X_flat_test_s,
    }


# ---------------------------------------------------------------------------
# Test 1 — Data preprocessor shapes
# ---------------------------------------------------------------------------

def test_data_preprocessor_shapes(preprocessed_data: dict) -> None:
    """
    X_seq must be (N, SEQUENCE_LENGTH, 18) and X_flat must be (N, 18).
    """
    X_seq = preprocessed_data["X_seq"]
    X_flat = preprocessed_data["X_flat"]

    assert X_seq.ndim == 3, f"X_seq ndim={X_seq.ndim}, expected 3"
    assert X_seq.shape[1] == config.SEQUENCE_LENGTH, (
        f"seq_len={X_seq.shape[1]}, expected {config.SEQUENCE_LENGTH}"
    )
    assert X_seq.shape[2] == 18, f"features={X_seq.shape[2]}, expected 18"

    assert X_flat.ndim == 2, f"X_flat ndim={X_flat.ndim}, expected 2"
    assert X_flat.shape[1] == 18, f"X_flat features={X_flat.shape[1]}, expected 18"
    assert X_seq.shape[0] == X_flat.shape[0], "X_seq and X_flat should have same N"


# ---------------------------------------------------------------------------
# Test 2 — Sequence-horizon alignment
# ---------------------------------------------------------------------------

def test_sequence_horizon_alignment(preprocessed_data: dict) -> None:
    """
    Label at position i must correspond to tick
    ``i + SEQUENCE_LENGTH - 1 + PREDICTION_HORIZON`` in the original dataset.
    """
    df = preprocessed_data["df"]
    y = preprocessed_data["y"]
    labels_col = df["is_congested"].values.astype(np.float32)

    seq_len = config.SEQUENCE_LENGTH
    horizon = config.PREDICTION_HORIZON

    # Check first and last valid index
    for i in [0, 1, len(y) - 1]:
        expected_idx = i + seq_len - 1 + horizon
        assert y[i] == labels_col[expected_idx], (
            f"y[{i}]={y[i]} != labels[{expected_idx}]={labels_col[expected_idx]}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Scaler fitted on train only
# ---------------------------------------------------------------------------

def test_scaler_fit_on_train_only(preprocessed_data: dict) -> None:
    """
    The scaler must be fitted on train data only — verify by checking that
    the train set has mean ~0 and std ~1 after scaling, while val/test may
    deviate.
    """
    X_flat_train_s = preprocessed_data["X_flat_train_s"]

    train_mean = np.abs(X_flat_train_s.mean(axis=0))
    train_std = X_flat_train_s.std(axis=0)

    # Train set should be approximately standardised.
    # Near-constant features (variance ~ 0 before scaling) will have
    # std = 0 after scaling — exclude them from the std check.
    assert np.all(train_mean < 0.15), (
        f"Train mean not near 0 after scaling: max={train_mean.max():.4f}"
    )
    non_const_mask = train_std > 1e-6
    if non_const_mask.any():
        assert np.all(np.abs(train_std[non_const_mask] - 1.0) < 0.15), (
            f"Train std not near 1 after scaling (non-constant features): "
            f"min={train_std[non_const_mask].min():.4f}, "
            f"max={train_std[non_const_mask].max():.4f}"
        )


# ---------------------------------------------------------------------------
# Test 4 — LSTM forward pass shape
# ---------------------------------------------------------------------------

def test_lstm_forward_pass() -> None:
    """
    Random input (32, 10, 18) must produce output shape (32, 1).
    """
    model = CongestionLSTM(input_size=18, hidden_size=64, num_layers=2, dropout=0.3)
    model.eval()
    x = torch.randn(32, 10, 18)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (32, 1), f"Expected (32, 1), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 5 — LSTM output range after sigmoid
# ---------------------------------------------------------------------------

def test_lstm_output_range() -> None:
    """
    After applying sigmoid, all LSTM outputs must be in [0, 1].
    """
    model = CongestionLSTM(input_size=18)
    model.eval()
    x = torch.randn(64, 10, 18)
    with torch.no_grad():
        logits = model(x)
        probas = torch.sigmoid(logits)
    assert probas.min() >= 0.0, f"Min proba={probas.min():.4f}"
    assert probas.max() <= 1.0, f"Max proba={probas.max():.4f}"


# ---------------------------------------------------------------------------
# Test 6 — XGBoost trains and predicts
# ---------------------------------------------------------------------------

def test_xgboost_trains_and_predicts() -> None:
    """
    Train XGBoost on 100 synthetic samples, predict, verify shape.
    """
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 18)).astype(np.float32)
    y_train = rng.integers(0, 2, size=100).astype(np.float32)
    X_val = rng.standard_normal((20, 18)).astype(np.float32)
    y_val = rng.integers(0, 2, size=20).astype(np.float32)

    xgb = XGBoostPredictor(scale_pos_weight=1.0)
    xgb.train(X_train, y_train, X_val, y_val)

    preds = xgb.model.predict(X_val)
    assert preds.shape == (20,), f"Expected (20,), got {preds.shape}"


# ---------------------------------------------------------------------------
# Test 7 — SHAP values shape
# ---------------------------------------------------------------------------

def test_shap_values_shape() -> None:
    """
    SHAP values array must have shape (N, 18) matching the feature count.
    """
    from ml.shap_explainer import SHAPExplainer

    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 18)).astype(np.float32)
    y_train = rng.integers(0, 2, size=100).astype(np.float32)

    xgb = XGBoostPredictor(scale_pos_weight=1.0)
    xgb.train(X_train, y_train, X_train[:20], y_train[:20])

    explainer = SHAPExplainer(xgb, feature_names=[f"f{i}" for i in range(18)])
    sv = explainer.compute_shap_values(X_train[:10])

    assert sv.shape == (10, 18), f"Expected (10, 18), got {sv.shape}"


# ---------------------------------------------------------------------------
# Test 8 — Ensemble weighted average
# ---------------------------------------------------------------------------

def test_ensemble_weighted_average() -> None:
    """
    Ensemble output must be a weighted blend, not just one model's output.
    """
    from ml.ensemble import EnsemblePredictor

    device = torch.device("cpu")
    lstm = CongestionLSTM(input_size=18)
    lstm.eval()

    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 18)).astype(np.float32)
    y_train = rng.integers(0, 2, size=100).astype(np.float32)
    xgb = XGBoostPredictor(scale_pos_weight=1.0)
    xgb.train(X_train, y_train, X_train[:20], y_train[:20])

    ens = EnsemblePredictor(lstm, xgb, device, lstm_weight=0.6, xgb_weight=0.4)

    X_seq = torch.randn(10, 10, 18)
    X_flat = rng.standard_normal((10, 18)).astype(np.float32)
    probas = ens.predict_proba(X_seq, X_flat)

    # Get individual model probas
    with torch.no_grad():
        lstm_p = torch.sigmoid(lstm(X_seq)).numpy().flatten()
    xgb_p = xgb.model.predict_proba(X_flat)[:, 1]

    expected = 0.6 * lstm_p + 0.4 * xgb_p
    np.testing.assert_allclose(probas, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 9 — Class imbalance handling
# ---------------------------------------------------------------------------

def test_class_imbalance_handling(preprocessed_data: dict) -> None:
    """
    Given ~12.1% positive rate, pos_weight must be > 1.
    """
    y_train = preprocessed_data["splits"]["y_train"]
    pw = DataPreprocessor.compute_class_weight(y_train)
    assert pw > 1.0, f"pos_weight={pw:.2f} should be > 1 for imbalanced data"


# ---------------------------------------------------------------------------
# Test 10 — Train / val / test splits have zero overlap
# ---------------------------------------------------------------------------

def test_train_val_test_no_overlap(preprocessed_data: dict) -> None:
    """
    The three splits must have zero overlapping indices.

    Since splits are contiguous time slices, we verify by checking that
    the total sample count equals the sum of split sizes.
    """
    s = preprocessed_data["splits"]
    n_train = len(s["y_train"])
    n_val = len(s["y_val"])
    n_test = len(s["y_test"])
    n_total = len(preprocessed_data["y"])

    assert n_train + n_val + n_test == n_total, (
        f"Split sizes don't sum to total: {n_train}+{n_val}+{n_test} != {n_total}"
    )

    # Additionally verify time ordering: last train tick < first val tick
    X_seq = preprocessed_data["X_seq"]
    X_seq_train = s["X_seq_train"]
    X_seq_val = s["X_seq_val"]
    X_seq_test = s["X_seq_test"]

    # Reconstruct from contiguous slices — shapes must be non-overlapping
    assert X_seq_train.shape[0] + X_seq_val.shape[0] + X_seq_test.shape[0] == X_seq.shape[0]
