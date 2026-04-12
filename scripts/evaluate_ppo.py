"""
scripts/evaluate_ppo.py

Evaluate a trained PPO agent with training_mode=False (no synthetic load spikes).
Runs baseline evaluator for comparison and writes phase5_results_v2.json matching
reports/phase5_results.json schema.

Usage from project root (5g-network-digital-twin):
    python scripts/evaluate_ppo.py
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from simulation.engine import NetworkSimulation
from ml.lstm_model import CongestionLSTM
from ml.xgboost_model import XGBoostPredictor
from ml.ensemble import EnsemblePredictor
from optimizer.rl_env import NetworkOptimizationEnv
from optimizer.agent import PPOAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_ppo")

Path("reports").mkdir(exist_ok=True)

BASELINE_V2 = "reports/baseline_results_v2.json"
PHASE5_V2 = "reports/phase5_results_v2.json"
SUMMARY_PATH = "reports/ppo_training_summary.json"


def _san(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _san(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_san(v) for v in obj]
    return obj


def _phase5_baseline_shape(full: dict) -> dict:
    episodes_data = []
    for e in full.get("episodes_data", []):
        episodes_data.append({
            "episode": e["episode"],
            "total_reward": e["total_reward"],
            "ticks": e["ticks"],
            "congestion_rate": e["congestion_rate"],
            "handovers_per_tick": e["handovers_per_tick"],
        })
    return {
        "mean_reward": full["mean_reward"],
        "std_reward": full["std_reward"],
        "mean_congestion_rate": full["mean_congestion_rate"],
        "mean_handovers_per_tick": full["mean_handovers_per_tick"],
        "episodes_data": episodes_data,
    }


def _phase5_ppo_shape(ppo: dict) -> dict:
    return _phase5_baseline_shape(ppo)


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
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    logger.info("Working directory: %s", root)

    py = sys.executable
    baseline_cmd = [
        py,
        str(root / "scripts" / "baseline_eval.py"),
        "--out",
        BASELINE_V2,
        "--episodes",
        "5",
    ]
    logger.info("Running baseline: %s", " ".join(baseline_cmd))
    subprocess.run(baseline_cmd, check=True, cwd=str(root))

    with open(BASELINE_V2, encoding="utf-8") as f:
        baseline_full = json.load(f)

    device = torch.device("cpu")
    ensemble = load_ensemble(device)
    sim = NetworkSimulation()
    env = NetworkOptimizationEnv(
        sim,
        ensemble,
        device,
        training_mode=False,
    )
    agent = PPOAgent(env, device="cpu")
    agent.load("models/ppo_agent")
    ppo_results = agent.evaluate(n_episodes=5)

    training_block: dict = {
        "total_timesteps": 200_000,
        "save_path": "models/ppo_agent",
        "eval_freq": 5_000,
    }
    if Path(SUMMARY_PATH).is_file():
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary = json.load(f)
        training_block.update(summary)

    report = {
        "baseline": _san(_phase5_baseline_shape(baseline_full)),
        "ppo_agent": _san(_phase5_ppo_shape(ppo_results)),
        "training": _san(training_block),
    }

    with open(PHASE5_V2, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Wrote %s", PHASE5_V2)
    print(f"Saved {PHASE5_V2} (baseline detail: {BASELINE_V2})")


if __name__ == "__main__":
    main()
