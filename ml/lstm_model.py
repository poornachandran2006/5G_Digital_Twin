"""
ml/lstm_model.py

LSTM-based congestion predictor and training harness for the 5G Digital Twin.

Architecture: LSTM -> BatchNorm -> FC(64->32) -> ReLU -> Dropout -> FC(32->1)
Training: BCEWithLogitsLoss with pos_weight, Adam + ReduceLROnPlateau,
          early stopping on validation F1.

Run from project root: python -m ml.lstm_model
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CongestionLSTM(nn.Module):
    """
    LSTM-based binary classifier for 5G cell congestion prediction.

    Architecture::

        LSTM(input_size, hidden_size, num_layers, dropout) ->
        BatchNorm1d(hidden_size) ->
        Linear(hidden_size, 32) -> ReLU -> Dropout(0.3) ->
        Linear(32, 1)

    The ``forward`` method returns raw logits (not sigmoid) so that
    ``BCEWithLogitsLoss`` is used for numerical stability.
    """

    def __init__(
        self,
        input_size: int = 18,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            input_size: Number of features per timestep.
            hidden_size: LSTM hidden dimension.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate between LSTM layers and before the
                     output head.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Logits tensor of shape ``(batch, 1)``.
        """
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take the last timestep's hidden state
        last_hidden = lstm_out[:, -1, :]          # (batch, hidden_size)

        out = self.batch_norm(last_hidden)
        out = self.fc1(out)                        # (batch, 32)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)                        # (batch, 1)
        return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LSTMTrainer:
    """
    Training harness for :class:`CongestionLSTM`.

    Handles the full training loop including early stopping on validation
    F1, learning rate scheduling, and model checkpointing.
    """

    def __init__(
        self,
        model: CongestionLSTM,
        device: torch.device,
        pos_weight: float,
        lr: float = 1e-3,
    ) -> None:
        """
        Args:
            model: The :class:`CongestionLSTM` instance.
            device: ``torch.device`` to train on (``cpu`` or ``cuda``).
            pos_weight: Positive-class weight for ``BCEWithLogitsLoss``.
            lr: Initial learning rate for Adam.
        """
        self.model = model.to(device)
        self.device = device

        pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5,
        )
        self.history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "val_f1": [], "val_auc": [],
        }

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Run one training epoch.

        Args:
            loader: Training :class:`DataLoader`.

        Returns:
            Tuple of (average loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, loader: DataLoader) -> dict:
        """
        Evaluate on a data loader and return comprehensive metrics.

        Args:
            loader: Validation or test :class:`DataLoader`.

        Returns:
            dict with keys: ``loss``, ``accuracy``, ``precision``, ``recall``,
            ``f1``, ``auc_roc``, ``confusion_matrix``.
        """
        self.model.eval()
        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        total_loss = 0.0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                total += X_batch.size(0)

                all_logits.append(logits.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        logits_np = np.concatenate(all_logits).flatten()
        labels_np = np.concatenate(all_labels).flatten()
        probas = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid
        labels_int = labels_np.astype(int)

        # Find threshold that maximises F1
        best_f1, best_thresh = 0.0, 0.5
        for t in np.arange(0.10, 0.90, 0.01):
            p = (probas >= t).astype(int)
            f = float(f1_score(labels_int, p, zero_division=0))
            if f > best_f1:
                best_f1, best_thresh = f, t

        preds = (probas >= best_thresh).astype(int)
        logger.info("LSTM optimal threshold: %.2f (F1=%.4f)", best_thresh, best_f1)

        return {
            "loss": total_loss / total,
            "accuracy": float(accuracy_score(labels_int, preds)),
            "precision": float(precision_score(labels_int, preds, zero_division=0)),
            "recall": float(recall_score(labels_int, preds, zero_division=0)),
            "f1": float(f1_score(labels_int, preds, zero_division=0)),
            "auc_roc": float(roc_auc_score(labels_int, probas)),
            "confusion_matrix": confusion_matrix(labels_int, preds).tolist(),
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10,
        model_path: str = "models/lstm_best.pt",
        history_path: str = "models/lstm_history.json",
    ) -> dict:
        """
        Full training loop with early stopping on validation F1.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Maximum number of epochs.
            patience: Early-stopping patience (epochs without F1 improvement).
            model_path: Path to save the best model checkpoint.
            history_path: Path to save the training history JSON.

        Returns:
            dict: Best validation metrics.
        """
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        best_f1 = -1.0
        wait = 0
        best_metrics: dict = {}

        header = (
            f"{'Ep':>3} | {'TrainLoss':>9} | {'TrainAcc':>8} | "
            f"{'ValLoss':>8} | {'ValAcc':>7} | {'ValPrec':>7} | "
            f"{'ValRec':>6} | {'ValF1':>6} | {'AUC':>6} | {'LR':>8}"
        )
        logger.info(header)
        logger.info("-" * len(header))

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            val_f1 = val_metrics["f1"]
            self.scheduler.step(val_f1)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_f1)
            self.history["val_auc"].append(val_metrics["auc_roc"])

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"{epoch:3d} | {train_loss:9.4f} | {train_acc:8.4f} | "
                f"{val_metrics['loss']:8.4f} | {val_metrics['accuracy']:7.4f} | "
                f"{val_metrics['precision']:7.4f} | {val_metrics['recall']:6.4f} | "
                f"{val_f1:6.4f} | {val_metrics['auc_roc']:6.4f} | {lr:8.6f}"
            )

            # Early stopping on val F1
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_metrics = val_metrics
                wait = 0
                torch.save(self.model.state_dict(), model_path)
                logger.info("  -> New best F1=%.4f, model saved", best_f1)
            else:
                wait += 1
                if wait >= patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                    break

        # Save history
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("Training history saved to %s", history_path)

        return best_metrics
