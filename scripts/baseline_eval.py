"""
scripts/baseline_eval.py

Baseline evaluation: run 5 episodes with action=0 (no-op) at every tick.

Computes: congestion_rate, avg_cell_load, peak_load, handovers_total.
Saves to reports/baseline_results.json as a comparison benchmark for PPO.

Usage from project root:
    python scripts/baseline_eval.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline_eval")

Path("reports").mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline (no-op) evaluation for Phase 5")
    parser.add_argument(
        "--out",
        default="reports/baseline_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--training-mode",
        action="store_true",
        help="Enable synthetic congestion injection (default: off, real-world-style loads)",
    )
    args = parser.parse_args()

    n_episodes = args.episodes
    device = torch.device("cpu")

    logger.info(
        "Loading models for baseline evaluation (training_mode=%s)",
        args.training_mode,
    )
    lstm = CongestionLSTM(input_size=18, hidden_size=64, num_layers=2, dropout=0.3)
    lstm.load_state_dict(torch.load("models/lstm_best.pt", map_location=device, weights_only=True))
    lstm.eval()

    xgb = XGBoostPredictor()
    xgb.load("models/xgboost_model.json")

    ensemble = EnsemblePredictor(lstm, xgb, device)

    episodes_data: list[dict] = []

    for ep in range(n_episodes):
        sim = NetworkSimulation()
        env = NetworkOptimizationEnv(
            sim, ensemble, device, training_mode=args.training_mode
        )
        obs, _ = env.reset()

        done = False
        ep_loads_all: list[list[float]] = []
        ep_handovers = 0
        ep_congested = 0
        ep_ticks = 0
        ep_reward = 0.0

        while not done:
            obs, reward, terminated, truncated, info = env.step(0)  # no-op
            done = terminated or truncated

            loads = info.get("cell_loads", [0.0, 0.0, 0.0])
            ep_loads_all.append(loads)
            ep_handovers += info.get("handovers", 0)
            if any(l > 0.9 for l in loads):
                ep_congested += 1
            ep_ticks += 1
            ep_reward += float(reward)

        env.close()

        load_arr = np.array(ep_loads_all)
        ep_data = {
            "episode": ep + 1,
            "ticks": ep_ticks,
            "total_reward": ep_reward,
            "congestion_rate": ep_congested / max(ep_ticks, 1),
            "avg_cell_load": float(load_arr.mean()),
            "peak_load": float(load_arr.max()),
            "handovers_total": ep_handovers,
            "handovers_per_tick": ep_handovers / max(ep_ticks, 1),
        }
        episodes_data.append(ep_data)
        logger.info(
            "Episode %d: reward=%.1f cong=%.1f%% avg_load=%.3f peak=%.3f ho=%d",
            ep + 1, ep_reward,
            ep_data["congestion_rate"] * 100,
            ep_data["avg_cell_load"],
            ep_data["peak_load"],
            ep_handovers,
        )

    results = {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean([e["total_reward"] for e in episodes_data])),
        "std_reward": float(np.std([e["total_reward"] for e in episodes_data])),
        "mean_congestion_rate": float(np.mean([e["congestion_rate"] for e in episodes_data])),
        "mean_avg_cell_load": float(np.mean([e["avg_cell_load"] for e in episodes_data])),
        "mean_peak_load": float(np.mean([e["peak_load"] for e in episodes_data])),
        "mean_handovers_total": float(np.mean([e["handovers_total"] for e in episodes_data])),
        "mean_handovers_per_tick": float(np.mean([e["handovers_per_tick"] for e in episodes_data])),
        "episodes_data": episodes_data,
    }

    out_path = args.out
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Baseline results saved to %s", out_path)
    print(f"\nBaseline (no agent) over {n_episodes} episodes:")
    print(f"  Congestion rate:  {results['mean_congestion_rate']:.3f}")
    print(f"  Avg cell load:    {results['mean_avg_cell_load']:.3f}")
    print(f"  Peak load:        {results['mean_peak_load']:.3f}")
    print(f"  Mean reward:      {results['mean_reward']:.1f}")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
