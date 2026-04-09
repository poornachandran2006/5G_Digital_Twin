"""
scripts/train_models.py

End-to-end Phase 4 ML training pipeline for the 5G Digital Twin.

Steps:
  a. Load and preprocess data
  b. Train LSTM with per-epoch metrics table
  c. Train XGBoost
  d. Evaluate both on test set, print comparison table
  e. Run SHAP on test set, save plots
  f. Evaluate ensemble on test set
  g. Save final metrics to reports/phase4_results.json

Usage from project root:
    python scripts/train_models.py
"""

from __future__ import annotations

import json
import logging
import sys
import os
from pathlib import Path

# Ensure project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from ml.data_preprocessor import DataPreprocessor
from ml.lstm_model import CongestionLSTM, LSTMTrainer
from ml.xgboost_model import XGBoostPredictor
from ml.shap_explainer import SHAPExplainer
from ml.ensemble import EnsemblePredictor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_models")


def _print_comparison(results: dict[str, dict]) -> None:
    """Print a side-by-side metrics comparison table."""
    names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]

    header = f"{'Metric':<12}" + "".join(f"{n:>14}" for n in names)
    print("\n" + "=" * len(header))
    print("  MODEL COMPARISON — Test Set")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics:
        row = f"{m:<12}" + "".join(f"{results[n].get(m, 0):>14.4f}" for n in names)
        print(row)
    print("=" * len(header) + "\n")


def main() -> None:
    """Execute the full Phase 4 training pipeline."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # ==================================================================
    # a. Data loading and preprocessing
    # ==================================================================
    logger.info("=" * 55)
    logger.info("  Step 1/7: Loading and preprocessing data")
    logger.info("=" * 55)

    prep = DataPreprocessor()
    df = prep.load_data("data/kpi_dataset.csv")

    X_seq, y, X_flat = prep.create_sequences(df)
    splits = prep.split_data(X_seq, X_flat, y)

    # Scale flat features (for XGBoost)
    X_flat_train_s, X_flat_val_s, X_flat_test_s = prep.scale_features(
        splits["X_flat_train"], splits["X_flat_val"], splits["X_flat_test"],
    )

    # Scale sequence features (for LSTM)
    X_seq_train_s, X_seq_val_s, X_seq_test_s = prep.scale_sequences(
        splits["X_seq_train"], splits["X_seq_val"], splits["X_seq_test"],
    )

    y_train = splits["y_train"]
    y_val = splits["y_val"]
    y_test = splits["y_test"]

    # Class weight
    pos_weight = prep.compute_class_weight(y_train)

    # DataLoaders for LSTM
    batch_size = 128

    train_ds = TensorDataset(
        torch.from_numpy(X_seq_train_s), torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_seq_val_s), torch.from_numpy(y_val),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_seq_test_s), torch.from_numpy(y_test),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ==================================================================
    # b. Train LSTM
    # ==================================================================
    logger.info("=" * 55)
    logger.info("  Step 2/7: Training LSTM")
    logger.info("=" * 55)

    lstm_model = CongestionLSTM(input_size=18, hidden_size=64, num_layers=2, dropout=0.3)
    trainer = LSTMTrainer(lstm_model, device, pos_weight=pos_weight)
    best_val_metrics = trainer.train(train_loader, val_loader, epochs=50, patience=10)

    # Reload best model for evaluation
    lstm_model.load_state_dict(torch.load("models/lstm_best.pt", map_location=device, weights_only=True))
    lstm_model = lstm_model.to(device)

    # ==================================================================
    # c. Train XGBoost
    # ==================================================================
    logger.info("=" * 55)
    logger.info("  Step 3/7: Training XGBoost")
    logger.info("=" * 55)

    xgb_pred = XGBoostPredictor(scale_pos_weight=pos_weight)
    xgb_pred.train(
        X_flat_train_s, y_train,
        X_flat_val_s, y_val,
        feature_names=prep.feature_names,
    )
    xgb_pred.save("models/xgboost_model.json")

    # ==================================================================
    # d. Evaluate both on test set
    # ==================================================================
    logger.info("=" * 55)
    logger.info("  Step 4/7: Evaluating models on test set")
    logger.info("=" * 55)

    lstm_test_metrics = trainer.evaluate(test_loader)
    xgb_test_metrics = xgb_pred.evaluate(X_flat_test_s, y_test)

    logger.info("LSTM test: F1=%.4f AUC=%.4f", lstm_test_metrics["f1"], lstm_test_metrics["auc_roc"])
    logger.info("XGB  test: F1=%.4f AUC=%.4f", xgb_test_metrics["f1"], xgb_test_metrics["auc_roc"])

    all_results: dict[str, dict] = {
        "LSTM": lstm_test_metrics,
        "XGBoost": xgb_test_metrics,
    }

    # ==================================================================
    # e. SHAP analysis
    # ==================================================================
    logger.info("=" * 55)
    logger.info("  Step 5/7: Running SHAP explainability")
    logger.info("=" * 55)

    # Use a subset for SHAP to keep runtime reasonable
    shap_sample_size = min(500, len(X_flat_test_s))
    X_shap = X_flat_test_s[:shap_sample_size]

    explainer = SHAPExplainer(xgb_pred, feature_names=prep.feature_names)
    explainer.plot_summary(X_shap, save_path="reports/shap_summary.png")
    explainer.plot_waterfall(X_shap, sample_idx=0, save_path="reports/shap_waterfall.png")
    top_features = explainer.get_top_features(X_shap, top_n=5)

    logger.info("Top 5 features by SHAP: %s", top_features)

    # ==================================================================
    # f. Ensemble evaluation
    # ==================================================================
    logger.info("=" * 55)
    logger.info("  Step 6/7: Evaluating ensemble")
    logger.info("=" * 55)

    ensemble = EnsemblePredictor(
        lstm_model, xgb_pred, device, lstm_weight=0.6, xgb_weight=0.4,
    )

    X_seq_test_tensor = torch.from_numpy(X_seq_test_s)
    ensemble_metrics = ensemble.evaluate(X_seq_test_tensor, X_flat_test_s, y_test)
    all_results["Ensemble"] = ensemble_metrics

    _print_comparison(all_results)

    # ==================================================================
    # g. Save results
    # ==================================================================
    logger.info("=" * 55)
    logger.info("  Step 7/7: Saving final results")
    logger.info("=" * 55)

    # Convert numpy types for JSON serialization
    def _sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    final_report = {
        "lstm_test": _sanitize(lstm_test_metrics),
        "xgboost_test": _sanitize(xgb_test_metrics),
        "ensemble_test": _sanitize(ensemble_metrics),
        "top_shap_features": [(n, float(v)) for n, v in top_features],
        "pos_weight": float(pos_weight),
        "dataset_shape": list(df.shape),
    }

    report_path = "reports/phase4_results.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    logger.info("Final report saved to %s", report_path)

    print("\n" + "=" * 55)
    print("  Phase 4 COMPLETE")
    print("=" * 55)
    print(f"  LSTM   F1={lstm_test_metrics['f1']:.4f}  AUC={lstm_test_metrics['auc_roc']:.4f}")
    print(f"  XGB    F1={xgb_test_metrics['f1']:.4f}  AUC={xgb_test_metrics['auc_roc']:.4f}")
    print(f"  ENS    F1={ensemble_metrics['f1']:.4f}  AUC={ensemble_metrics['auc_roc']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
