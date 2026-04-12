"""
scripts/train_ppo.py

Train PPO with congestion injection (training_mode=True), 200k timesteps,
file + console logging, and rollout metrics every 2048 steps.

Usage from project root (5g-network-digital-twin):
    python scripts/train_ppo.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from simulation.engine import NetworkSimulation
from ml.lstm_model import CongestionLSTM
from ml.xgboost_model import XGBoostPredictor
from ml.ensemble import EnsemblePredictor
from optimizer.rl_env import NetworkOptimizationEnv
from optimizer.agent import PPOAgent


REPORTS_DIR = Path("reports")
LOG_PATH = REPORTS_DIR / "ppo_training.log"
SUMMARY_PATH = REPORTS_DIR / "ppo_training_summary.json"
TOTAL_TIMESTEPS = 200_000
ROLLOUT_LOG_INTERVAL = 2048


def _setup_logging() -> logging.Logger:
    REPORTS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("train_ppo")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


class PPOTrainingMetricsCallback(BaseCallback):
    """
    On each rollout end (n_steps timesteps), log mean reward, SB3 losses,
    and action histogram for the last rollout. Accumulates global action counts.
    """

    def __init__(self, log: logging.Logger, log_every: int = ROLLOUT_LOG_INTERVAL):
        super().__init__(verbose=0)
        self._log = log
        self.log_every = log_every
        self.last_mean_reward: float = 0.0
        self.action_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        buf = self.model.rollout_buffer
        rewards = np.asarray(buf.rewards).squeeze()
        self.last_mean_reward = float(np.mean(rewards))

        actions = buf.actions
        if actions.ndim > 1:
            flat = actions.reshape(-1)
        else:
            flat = np.asarray(actions).reshape(-1)
        for a in flat:
            ai = int(a)
            self.action_counts[ai] = self.action_counts.get(ai, 0) + 1

        vals = dict(self.model.logger.name_to_value)
        entropy_loss = vals.get("train/entropy_loss", vals.get("entropy_loss", float("nan")))
        pg_loss = vals.get(
            "train/policy_gradient_loss",
            vals.get("policy_gradient_loss", vals.get("train/loss", float("nan"))),
        )
        value_loss = vals.get("train/value_loss", vals.get("value_loss", float("nan")))

        hist = {str(k): int(v) for k, v in sorted(self.action_counts.items())}
        rollout_hist: dict[str, int] = {}
        for a in flat:
            rollout_hist[str(int(a))] = rollout_hist.get(str(int(a)), 0) + 1

        self._log.info(
            "Rollout @ %d timesteps | mean_reward=%.4f | entropy_loss=%s | "
            "policy_loss(pg)=%s | value_loss=%s | rollout_actions=%s | cum_actions=%s",
            self.num_timesteps,
            self.last_mean_reward,
            entropy_loss,
            pg_loss,
            value_loss,
            rollout_hist,
            hist,
        )
        return True


def load_ensemble(device: torch.device) -> EnsemblePredictor:
    lstm = CongestionLSTM(input_size=18, hidden_size=64, num_layers=2, dropout=0.3)
    lstm.load_state_dict(
        torch.load("models/lstm_best.pt", map_location=device, weights_only=True)
    )
    lstm.eval()
    xgb = XGBoostPredictor()
    xgb.load("models/xgboost_model.json")
    return EnsemblePredictor(lstm, xgb, device)


def main() -> None:
    log = _setup_logging()
    Path("models").mkdir(exist_ok=True)

    device = torch.device("cpu")
    log.info("PPO training starting: %d timesteps, training_mode=congestion_injection", TOTAL_TIMESTEPS)

    ensemble = load_ensemble(device)
    sim = NetworkSimulation()
    env = NetworkOptimizationEnv(
        sim,
        ensemble,
        device,
        training_mode=True,
    )

    log.info("Running SB3 env check")
    check_env(env, warn=True)

    metrics_cb = PPOTrainingMetricsCallback(log, log_every=ROLLOUT_LOG_INTERVAL)
    agent = PPOAgent(env, device="cpu")
    agent.train(
        total_timesteps=TOTAL_TIMESTEPS,
        save_path="models/ppo_agent",
        eval_freq=5_000,
        extra_callbacks=[metrics_cb],
    )

    total_actions = sum(metrics_cb.action_counts.values())
    if total_actions > 0:
        action_percentages = {
            k: round(100.0 * metrics_cb.action_counts.get(k, 0) / total_actions, 4)
            for k in (0, 1, 2, 3)
        }
    else:
        action_percentages = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    summary = {
        "total_timesteps": TOTAL_TIMESTEPS,
        "final_mean_reward": float(metrics_cb.last_mean_reward),
        "action_counts": {k: metrics_cb.action_counts.get(k, 0) for k in (0, 1, 2, 3)},
        "action_percentages": {k: action_percentages[k] for k in (0, 1, 2, 3)},
        "training_mode": "congestion_injection",
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("Training complete. Summary written to %s", SUMMARY_PATH)
    print(f"Saved {SUMMARY_PATH} and models/ppo_agent.zip")


if __name__ == "__main__":
    main()
