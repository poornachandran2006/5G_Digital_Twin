"""
optimizer/rule_based_agent.py

Rule-based load balancing baseline for A/B comparison against PPO.

Policy:
  - If any cell load > 0.9 (critical): emergency handover (mirrors action 3)
  - Elif any cell load > 0.8 (warning): mass balance (mirrors action 2)
  - Else: do nothing (mirrors action 0)

This is intentionally simple — the point is to show PPO learns
something a naive threshold policy cannot.
"""

from __future__ import annotations

import numpy as np

_LOAD_WARNING  = 0.80   # slightly tighter than PPO env's 0.70 to be "reasonable"
_LOAD_CRITICAL = 0.90
_LOAD_HEALTHY  = 0.50


class RuleBasedAgent:
    """
    Threshold-based load balancer. No training, no ML.
    Takes the same 9-dim observation as PPO and returns an action int.
    """

    def __init__(self) -> None:
        self.total_reward: float = 0.0
        self.tick_count: int = 0
        self.action_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

    def predict(self, obs: np.ndarray) -> int:
        """
        obs: shape (9,) — [load0, load1, load2, cong0, cong1, cong2, ue0, ue1, ue2]
        Returns action int: 0=noop, 1=balance, 2=mass_balance, 3=emergency
        """
        loads = obs[:3]  # first 3 dims are cell loads

        # Emergency: any cell critically overloaded
        if np.any(loads > _LOAD_CRITICAL):
            action = 3
        # Warning: any cell above warning threshold
        elif np.any(loads > _LOAD_WARNING):
            action = 2
        # Imbalanced: max - min spread too large
        elif (loads.max() - loads.min()) > 0.30:
            action = 1
        else:
            action = 0

        self.action_counts[action] += 1
        return action

    def record_reward(self, obs: np.ndarray) -> float:
        """
        Compute the same reward formula used in rl_env.py step().
        Call this AFTER the environment steps with the rule-based action.
        obs: the NEW observation after the action was applied.
        """
        loads = obs[:3]
        reward = 0.0
        for load in loads:
            if load < 0.70:
                reward += 0.2
            elif load < 0.90:
                reward -= 0.5
            else:
                reward -= 1.0

        balance_bonus = max(0.0, 0.1 - float(np.std(loads)))
        reward += balance_bonus
        reward = float(np.clip(reward, -1.0, 1.0))

        self.total_reward += reward
        self.tick_count += 1
        return reward

    def get_stats(self) -> dict:
        avg = self.total_reward / max(self.tick_count, 1)
        return {
            "total_reward": round(self.total_reward, 3),
            "avg_reward": round(avg, 4),
            "tick_count": self.tick_count,
            "action_counts": self.action_counts.copy(),
        }

    def reset_stats(self) -> None:
        self.total_reward = 0.0
        self.tick_count = 0
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}