"""
scripts/train_rl_agent.py

End-to-end Phase 5 RL training pipeline for the 5G Digital Twin.

Steps:
  a. Load simulation engine
  b. Load EnsemblePredictor from saved models
  c. Instantiate NetworkOptimizationEnv
  d. Validate env with SB3 check_env
  e. Instantiate and train PPOAgent (50 000 steps)
  f. Evaluate for 5 episodes
  g. Compare against baseline (no-agent)
  h. Save results to reports/phase5_results.json

Usage from project root:
    python scripts/train_rl_agent.py
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from stable_baselines3.common.env_checker import check_env

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
logger = logging.getLogger("train_rl_agent")

Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)


def load_ensemble(device: torch.device) -> EnsemblePredictor:
    """Load and return the trained EnsemblePredictor from saved model files."""
    logger.info("Loading LSTM from models/lstm_best.pt")
    lstm = CongestionLSTM(input_size=18, hidden_size=64, num_layers=2, dropout=0.3)
    lstm.load_state_dict(torch.load("models/lstm_best.pt", map_location=device, weights_only=True))
    lstm.eval()

    logger.info("Loading XGBoost from models/xgboost_model.json")
    xgb = XGBoostPredictor()
    xgb.load("models/xgboost_model.json")

    return EnsemblePredictor(lstm, xgb, device)


def run_baseline(n_episodes: int = 5) -> dict:
    """
    Run baseline (always no-op action=0) for *n_episodes* episodes.

    Returns:
        dict: Baseline metrics.
    """
    logger.info("Running baseline (no-agent) for %d episodes", n_episodes)
    device = torch.device("cpu")
    ensemble = load_ensemble(device)
    episodes_data = []

    for ep in range(n_episodes):
        sim = NetworkSimulation()
        env = NetworkOptimizationEnv(sim, ensemble, device)
        obs, _ = env.reset()
        done = False
        ep_reward, ep_ticks, ep_congested, ep_handovers = 0.0, 0, 0, 0

        while not done:
            obs, reward, terminated, truncated, info = env.step(0)  # always no-op
            done = terminated or truncated
            ep_reward += float(reward)
            ep_ticks += 1
            loads = info.get("cell_loads", [0.0, 0.0, 0.0])
            if any(l > 0.9 for l in loads):
                ep_congested += 1
            ep_handovers += info.get("handovers", 0)

        env.close()

        episodes_data.append({
            "episode": ep + 1,
            "total_reward": ep_reward,
            "ticks": ep_ticks,
            "congestion_rate": ep_congested / max(ep_ticks, 1),
            "handovers_per_tick": ep_handovers / max(ep_ticks, 1),
        })

    import numpy as np
    return {
        "mean_reward": float(np.mean([e["total_reward"] for e in episodes_data])),
        "std_reward": float(np.std([e["total_reward"] for e in episodes_data])),
        "mean_congestion_rate": float(np.mean([e["congestion_rate"] for e in episodes_data])),
        "mean_handovers_per_tick": float(np.mean([e["handovers_per_tick"] for e in episodes_data])),
        "episodes_data": episodes_data,
    }


def _print_comparison(baseline: dict, ppo: dict) -> None:
    """Print a side-by-side comparison table."""
    metrics = [
        ("Avg Reward", "mean_reward", ".1f"),
        ("Std Reward", "std_reward", ".1f"),
        ("Congestion Rate", "mean_congestion_rate", ".3f"),
        ("Handovers/Tick", "mean_handovers_per_tick", ".2f"),
    ]
    header = f"{'Metric':<22} {'Baseline':>12} {'PPO Agent':>12}"
    print("\n" + "=" * len(header))
    print("  Phase 5 — Baseline vs PPO Agent")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label, key, fmt in metrics:
        bval = baseline.get(key, 0)
        pval = ppo.get(key, 0)
        print(f"{label:<22} {bval:>12{fmt}} {pval:>12{fmt}}")
    print("=" * len(header) + "\n")


def main() -> None:
    device = torch.device("cpu")
    logger.info("Phase 5 — RL training pipeline starting")

    # ------------------------------------------------------------------
    # a. Load simulation engine + ensemble
    # ------------------------------------------------------------------
    logger.info("Step 1/8: Creating simulation engine")
    sim = NetworkSimulation()

    logger.info("Step 2/8: Loading EnsemblePredictor")
    ensemble = load_ensemble(device)

    # ------------------------------------------------------------------
    # c. Instantiate environment
    # ------------------------------------------------------------------
    logger.info("Step 3/8: Creating NetworkOptimizationEnv")
    env = NetworkOptimizationEnv(sim, ensemble, device)

    # ------------------------------------------------------------------
    # d. Validate env
    # ------------------------------------------------------------------
    logger.info("Step 4/8: Running SB3 env check")
    check_env(env, warn=True)
    logger.info("check_env passed")

    # ------------------------------------------------------------------
    # e. Train PPO
    # ------------------------------------------------------------------
    logger.info("Step 5/8: Training PPO agent (50 000 timesteps)")
    agent = PPOAgent(env, device="cpu")
    train_info = agent.train(total_timesteps=50_000, save_path="models/ppo_agent")

    # ------------------------------------------------------------------
    # f. Evaluate PPO
    # ------------------------------------------------------------------
    logger.info("Step 6/8: Evaluating PPO for 5 episodes")
    ppo_results = agent.evaluate(n_episodes=5)

    # ------------------------------------------------------------------
    # g. Baseline comparison
    # ------------------------------------------------------------------
    logger.info("Step 7/8: Running baseline evaluation")
    baseline_results = run_baseline(n_episodes=2)  # 2 episodes to save time

    _print_comparison(baseline_results, ppo_results)

    # ------------------------------------------------------------------
    # h. Save results
    # ------------------------------------------------------------------
    logger.info("Step 8/8: Saving results to reports/phase5_results.json")

    def _san(obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _san(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_san(v) for v in obj]
        return obj

    report = {
        "baseline": _san(baseline_results),
        "ppo_agent": _san(ppo_results),
        "training": _san(train_info),
    }
    with open("reports/phase5_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Phase 5 COMPLETE")
    print(f"  PPO mean reward:      {ppo_results['mean_reward']:.1f}")
    print(f"  PPO congestion rate:  {ppo_results['mean_congestion_rate']:.3f}")
    print(f"  Saved: reports/phase5_results.json, models/ppo_agent.zip")


if __name__ == "__main__":
    main()
