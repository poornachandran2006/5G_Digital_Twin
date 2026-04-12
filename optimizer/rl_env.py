"""
optimizer/rl_env.py

Gymnasium environment wrapping the 5G Network Digital Twin simulation.

Observation: 9-dim float32 — cell loads, LSTM congestion probs, UE counts.
Action: Discrete(4). Rewards are clipped to [-1, 1] per step for stable PPO value learning.

Congestion spikes (training_mode=True): every 500 ticks, one random cell is held at
85–95% synthetic load for 100 ticks for the observation/reward signal.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

import config
from simulation.engine import NetworkSimulation, SimulationState
from ml.ensemble import EnsemblePredictor

logger = logging.getLogger(__name__)

_LOAD_WARNING = 0.7
_LOAD_CRITICAL = 0.9
_LOAD_HEALTHY_LOW = 0.5


def _state_to_feature_row(state: SimulationState) -> np.ndarray:
    gs = state.gnb_states
    row = np.array([
        gs[0]["load"], gs[1]["load"], gs[2]["load"],
        sum(u["throughput_mbps"] for u in state.ue_states if u["serving_gnb_id"] == 0),
        sum(u["throughput_mbps"] for u in state.ue_states if u["serving_gnb_id"] == 1),
        sum(u["throughput_mbps"] for u in state.ue_states if u["serving_gnb_id"] == 2),
        sum(1 for u in state.ue_states if u["serving_gnb_id"] == 0),
        sum(1 for u in state.ue_states if u["serving_gnb_id"] == 1),
        sum(1 for u in state.ue_states if u["serving_gnb_id"] == 2),
        float(np.mean([u["sinr_db"] for u in state.ue_states if u["serving_gnb_id"] == 0] or [0])),
        float(np.mean([u["sinr_db"] for u in state.ue_states if u["serving_gnb_id"] == 1] or [0])),
        float(np.mean([u["sinr_db"] for u in state.ue_states if u["serving_gnb_id"] == 2] or [0])),
        sum(u["throughput_mbps"] for u in state.ue_states),
        state.avg_sinr_db,
        float(np.mean([1000.0 / max(u["throughput_mbps"], 0.1) for u in state.ue_states])),
        float(state.handover_count),
        state.handover_count / max(len(state.ue_states), 1),
        float(sum(1 for u in state.ue_states if u["sinr_db"] < (config.SINR_MIN_DB + 3.0))
              / max(len(state.ue_states), 1)),
    ], dtype=np.float32)
    return row


class NetworkOptimizationEnv(gym.Env):
    """Gymnasium env for 5G load-balancing RL."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        sim_engine: NetworkSimulation,
        ensemble_model: EnsemblePredictor,
        device: torch.device,
        sequence_buffer_size: int = 10,
        render_mode: str | None = None,
        congestion_injection_prob: float = 0.08,
        training_mode: bool = True,
    ) -> None:
        super().__init__()

        self._sim = sim_engine
        self._ensemble = ensemble_model
        self._device = device
        self._render_mode = render_mode
        self._congestion_injection_prob = congestion_injection_prob
        self.training_mode = training_mode

        self._buffer: deque[np.ndarray] = deque(maxlen=sequence_buffer_size)
        self._seq_len = sequence_buffer_size

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self._tick: int = 0
        self.current_tick: int = 0
        self.cell_loads: np.ndarray = np.zeros(3, dtype=np.float64)
        self.spike_cell: int = 0
        self.spike_remaining: int = 0

        self._state: SimulationState | None = None
        self._last_reward: float = 0.0
        self._sim_iter = None

        logger.info(
            "NetworkOptimizationEnv: obs=%s actions=%d training_mode=%s",
            self.observation_space.shape, self.action_space.n, self.training_mode,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._sim = NetworkSimulation()
        rng = np.random.default_rng(seed)
        for ue in self._sim.ues:
            ue.position = (
                float(rng.uniform(0, config.GRID_SIZE_M)),
                float(rng.uniform(0, config.GRID_SIZE_M)),
            )

        self._sim_iter = self._sim.run(ticks=config.SIM_DURATION_S)
        self._buffer.clear()
        self._tick = 0
        self.current_tick = 0
        self.spike_cell = 0
        self.spike_remaining = 0

        self._state = next(self._sim_iter)
        self._tick = self._state.tick + 1
        self._buffer.append(_state_to_feature_row(self._state))

        self.cell_loads = np.array(
            [self._state.gnb_states[i]["load"] for i in range(3)], dtype=np.float64,
        )

        obs = self._get_observation()
        info = {
            "tick": 0,
            "cell_loads": self.cell_loads.tolist(),
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None, "Call reset() before step()"

        self._apply_action(action)

        if not self.training_mode:
            self._maybe_inject_congestion()

        try:
            self._state = next(self._sim_iter)
        except StopIteration:
            terminated = True
            truncated = False
            obs = self._get_observation()
            return obs, 0.0, terminated, truncated, {"tick": self._tick}

        self._tick = self._state.tick + 1
        self._buffer.append(_state_to_feature_row(self._state))

        self.cell_loads = np.array(
            [self._state.gnb_states[i]["load"] for i in range(3)], dtype=np.float64,
        )

        if self.training_mode:
            if self.current_tick % 500 == 0:
                self.spike_cell = int(np.random.randint(0, 3))
                self.spike_remaining = 100
            if self.spike_remaining > 0:
                self.cell_loads[self.spike_cell] = float(np.random.uniform(0.85, 0.95))
                self.spike_remaining -= 1

        self.current_tick += 1

        obs = self._get_observation()

        reward = 0.0
        cell_loads = obs[:3]
        for load in cell_loads:
            if load < 0.70:
                reward += 0.2
            elif load < 0.90:
                reward -= 0.5
            else:
                reward -= 1.0

        balance_bonus = max(0.0, 0.1 - float(np.std(cell_loads)))
        reward += balance_bonus

        reward = float(np.clip(reward, -1.0, 1.0))
        self._last_reward = reward

        terminated = self._tick >= config.SIM_DURATION_S
        truncated = False

        info = {
            "tick": self._tick,
            "cell_loads": self.cell_loads.tolist(),
            "handovers": self._state.handover_count,
            "congestion_probs": self._get_congestion_probs().tolist(),
        }

        if self._render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self._render_mode != "human" or self._state is None:
            return
        probs = self._get_congestion_probs()
        print(
            f"Tick {self._tick:5d} | "
            f"Loads [{self.cell_loads[0]:.3f} {self.cell_loads[1]:.3f} {self.cell_loads[2]:.3f}] | "
            f"CongProb [{probs[0]:.2f} {probs[1]:.2f} {probs[2]:.2f}] | "
            f"Reward {self._last_reward:+.2f}"
        )

    def close(self) -> None:
        self._sim_iter = None
        self._state = None
        logger.info("NetworkOptimizationEnv closed")

    def _get_observation(self) -> np.ndarray:
        if self._state is None:
            return np.zeros(9, dtype=np.float32)

        loads = np.clip(self.cell_loads.astype(np.float32), 0.0, 1.0)
        cong_probs = self._get_congestion_probs()
        ue_counts = np.array(
            [sum(1 for u in self._state.ue_states if u["serving_gnb_id"] == i)
             for i in range(3)],
            dtype=np.float32,
        ) / float(config.NUM_UE)
        ue_counts = np.clip(ue_counts, 0.0, 1.0)
        return np.concatenate([loads, cong_probs, ue_counts]).astype(np.float32)

    def _get_congestion_probs(self) -> np.ndarray:
        if len(self._buffer) < self._seq_len:
            return np.zeros(3, dtype=np.float32)

        if hasattr(self, '_cong_cache_tick') and self._tick - self._cong_cache_tick < 5:
            return self._cong_cache

        seq = np.stack(list(self._buffer), axis=0)
        seq_tensor = torch.from_numpy(seq).unsqueeze(0)
        flat = seq[-1:, :]
        proba = self._ensemble.predict_proba(seq_tensor, flat)
        system_prob = float(proba[0])

        if self._state is None:
            return np.zeros(3, dtype=np.float32)

        loads = np.array(
            [self._state.gnb_states[i]["load"] for i in range(3)], dtype=np.float32,
        )
        total_load = loads.sum()
        if total_load > 0:
            cell_probs = (loads / total_load) * system_prob * 3.0
        else:
            cell_probs = np.zeros(3, dtype=np.float32)

        return np.clip(cell_probs, 0.0, 1.0).astype(np.float32)

    def _apply_action(self, action: int) -> None:
        if action == 0:
            return

        loads = [g["load"] for g in self._state.gnb_states]

        if action == 1:
            src = int(np.argmax(loads))
            dst = int(np.argmin(loads))
            if src == dst:
                return
            candidates = [u for u in self._sim.ues if u.serving_gnb_id == src]
            if candidates:
                self._sim.set_ue_override(candidates[0].ue_id, dst)

        elif action == 2:
            overloaded = [i for i, l in enumerate(loads) if l > _LOAD_WARNING]
            underloaded = [i for i, l in enumerate(loads) if l < _LOAD_HEALTHY_LOW]
            if not overloaded or not underloaded:
                return
            dst = underloaded[0]
            moved = 0
            for src in overloaded:
                if moved >= 3:
                    break
                candidates = [u for u in self._sim.ues if u.serving_gnb_id == src]
                for ue in candidates:
                    if moved >= 3:
                        break
                    self._sim.set_ue_override(ue.ue_id, dst)
                    moved += 1

        elif action == 3:
            dst = int(np.argmin(loads))
            critical = [i for i, l in enumerate(loads)
                        if l > _LOAD_CRITICAL and i != dst]
            for src in critical:
                for ue in self._sim.ues:
                    if ue.serving_gnb_id == src:
                        self._sim.set_ue_override(ue.ue_id, dst)

    def _maybe_inject_congestion(self) -> None:
        if self._state is None:
            return
        if self.np_random.random() >= self._congestion_injection_prob:
            return

        loads = [g["load"] for g in self._state.gnb_states]
        target_cell = int(np.argmax(loads))
        candidates = [u for u in self._sim.ues if u.serving_gnb_id != target_cell]
        if candidates:
            idx = int(self.np_random.integers(0, len(candidates)))
            self._sim.set_ue_override(candidates[idx].ue_id, target_cell)


__all__ = ["NetworkOptimizationEnv"]
