"""
optimizer/rl_env.py

Gymnasium environment wrapping the 5G Network Digital Twin simulation.

Observation: 9-dim float32 vector — cell loads, LSTM congestion probs, UE counts.
Action: Discrete(4) — no-op, soft rebalance, aggressive rebalance, emergency offload.
Reward: shaped around cell load targets with handover penalty and all-clear bonus.

Key design: RL actions are applied via engine.set_ue_override(), which survives
exactly one simulation tick before physics resumes. This prevents the physics engine
from silently overwriting the agent's decisions every tick.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

import config
from simulation.engine import NetworkSimulation, SimulationState
from ml.ensemble import EnsemblePredictor
from ml.data_preprocessor import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Reward constants
_R_DROP_BELOW_WARNING = 2.0
_R_HEALTHY = 1.0
_R_CRITICAL = -3.0
_R_WARNING = -1.5
_R_HANDOVER = -0.5
_R_ALL_CLEAR = 5.0
_LOAD_WARNING = 0.7
_LOAD_CRITICAL = 0.9
_LOAD_HEALTHY_LOW = 0.5


def _state_to_feature_row(state: SimulationState) -> np.ndarray:
    """
    Extract the 18-feature row from a SimulationState matching
    the order defined in FEATURE_COLUMNS.

    Args:
        state: Current simulation state snapshot.

    Returns:
        np.ndarray: Shape (18,) float32 feature vector.
    """
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
    """
    Gymnasium environment for 5G network load-balancing via RL.

    Observation space (9-dim Box[0,1]):
    - [0-2]: cell PRB loads, normalised to [0, 1]
    - [3-5]: per-cell LSTM congestion probability
    - [6-8]: per-cell UE count normalised by NUM_UE (20)

    Action space (Discrete(4)):
    - 0: no-op
    - 1: soft rebalance — move 1 UE from most-loaded to least-loaded cell
    - 2: aggressive rebalance — move up to 3 UEs from cells > 0.7 to cells < 0.5
    - 3: emergency offload — move ALL UEs from cells > 0.9 to least-loaded cell

    Actions are applied via engine.set_ue_override() which survives one full
    simulation tick before physics resumes. This is the correct integration
    — the agent's decision is not silently overwritten.

    Congestion injection: with probability congestion_injection_prob each tick,
    a random UE is temporarily spiked to a loaded cell to create realistic
    congestion events for the agent to handle.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        sim_engine: NetworkSimulation,
        ensemble_model: EnsemblePredictor,
        device: torch.device,
        sequence_buffer_size: int = 10,
        render_mode: str | None = None,
        congestion_injection_prob: float = 0.08,
    ) -> None:
        """
        Args:
            sim_engine: Pre-built NetworkSimulation instance.
            ensemble_model: Trained EnsemblePredictor.
            device: Torch device for LSTM inference.
            sequence_buffer_size: Length of observation history buffer for LSTM.
            render_mode: 'human' to print per-tick tables; None for silence.
            congestion_injection_prob: Probability per tick of injecting a load
                spike on a random cell. Creates realistic congestion for training.
        """
        super().__init__()

        self._sim = sim_engine
        self._ensemble = ensemble_model
        self._device = device
        self._render_mode = render_mode
        self._congestion_injection_prob = congestion_injection_prob

        self._buffer: deque[np.ndarray] = deque(maxlen=sequence_buffer_size)
        self._seq_len = sequence_buffer_size

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self._tick: int = 0
        self._state: SimulationState | None = None
        self._last_reward: float = 0.0
        self._sim_iter = None

        logger.info(
            "NetworkOptimizationEnv created: obs=%s action=%s seq_len=%d injection_prob=%.2f",
            self.observation_space.shape, self.action_space.n,
            self._seq_len, self._congestion_injection_prob,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the simulation to tick 0 and clear the observation buffer.

        Each episode starts with randomized UE positions to ensure variance
        across training episodes (std_reward > 0).

        Args:
            seed: Optional RNG seed for reproducibility.
            options: Ignored.

        Returns:
            Tuple of (observation shape(9,), info dict).
        """
        super().reset(seed=seed)

        # Rebuild simulation from scratch for a clean episode
        self._sim = NetworkSimulation()

        # FIX Bug 3: Randomize UE starting positions each episode.
        # This ensures episodes are not identical (std_reward > 0).
        rng = np.random.default_rng(seed)
        for ue in self._sim.ues:
            ue.position = (
                float(rng.uniform(0, config.GRID_SIZE_M)),
                float(rng.uniform(0, config.GRID_SIZE_M)),
            )

        self._sim_iter = self._sim.run(ticks=config.SIM_DURATION_S)
        self._buffer.clear()
        self._tick = 0

        # Advance one tick to get initial state
        self._state = next(self._sim_iter)
        self._tick = self._state.tick + 1
        self._buffer.append(_state_to_feature_row(self._state))

        obs = self._get_observation()
        info = {
            "tick": 0,
            "cell_loads": [g["load"] for g in self._state.gnb_states],
        }
        return obs, info

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply action, advance simulation by one tick, compute reward.

        Action is applied via set_ue_override() BEFORE the tick advances,
        so the physics engine respects the agent's UE assignment for this tick.
        On the next tick, clear_overrides() is called automatically and physics
        resumes.

        Args:
            action: Integer in {0, 1, 2, 3}.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        assert self._state is not None, "Call reset() before step()"

        prev_loads = [g["load"] for g in self._state.gnb_states]

        # FIX Bug 1: Apply action via set_ue_override() — survives one tick.
        # The engine clears overrides at the start of the NEXT tick.
        self._apply_action(action)

        # FIX Bug 2: Congestion injection — occasionally force a UE to an
        # already-loaded cell to create realistic congestion events.
        self._maybe_inject_congestion()

        # Advance simulation — overrides are active for this tick
        try:
            self._state = next(self._sim_iter)
        except StopIteration:
            terminated = True
            truncated = False
            obs = self._get_observation()
            return obs, 0.0, terminated, truncated, {"tick": self._tick}

        self._tick = self._state.tick + 1
        self._buffer.append(_state_to_feature_row(self._state))

        new_loads = [g["load"] for g in self._state.gnb_states]
        handovers = self._state.handover_count

        reward, breakdown = self._compute_reward(prev_loads, new_loads, handovers)
        self._last_reward = reward

        obs = self._get_observation()
        terminated = self._tick >= config.SIM_DURATION_S
        truncated = False

        info = {
            "tick": self._tick,
            "cell_loads": new_loads,
            "handovers": handovers,
            "congestion_probs": self._get_congestion_probs().tolist(),
            "reward_breakdown": breakdown,
        }

        if self._render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Print a formatted table of current tick state (human render mode)."""
        if self._render_mode != "human" or self._state is None:
            return
        loads = [g["load"] for g in self._state.gnb_states]
        probs = self._get_congestion_probs()
        print(
            f"Tick {self._tick:5d} | "
            f"Loads [{loads[0]:.3f} {loads[1]:.3f} {loads[2]:.3f}] | "
            f"CongProb [{probs[0]:.2f} {probs[1]:.2f} {probs[2]:.2f}] | "
            f"Reward {self._last_reward:+.2f}"
        )

    def close(self) -> None:
        """Clean up environment resources."""
        self._sim_iter = None
        self._state = None
        logger.info("NetworkOptimizationEnv closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """
        Build the 9-dim observation vector from the current state.

        Returns:
            np.ndarray: shape(9,) float32, all values in [0, 1].
        """
        if self._state is None:
            return np.zeros(9, dtype=np.float32)

        gs = self._state.gnb_states
        loads = np.array([gs[i]["load"] for i in range(3)], dtype=np.float32)
        loads = np.clip(loads, 0.0, 1.0)

        cong_probs = self._get_congestion_probs()

        ue_counts = np.array(
            [sum(1 for u in self._state.ue_states if u["serving_gnb_id"] == i)
             for i in range(3)],
            dtype=np.float32,
        ) / float(config.NUM_UE)
        ue_counts = np.clip(ue_counts, 0.0, 1.0)

        obs = np.concatenate([loads, cong_probs, ue_counts]).astype(np.float32)
        return obs

    def _get_congestion_probs(self) -> np.ndarray:
        """
        Run EnsemblePredictor on the current observation buffer.

        Returns:
            np.ndarray: Shape (3,) — one probability per cell.
                        Returns zeros(3) if buffer has fewer than seq_len entries.
        """
        if len(self._buffer) < self._seq_len:
            return np.zeros(3, dtype=np.float32)

        if hasattr(self, '_cong_cache_tick') and self._tick - self._cong_cache_tick < 5:
            return self._cong_cache

        seq = np.stack(list(self._buffer), axis=0)       # (seq_len, 18)
        seq_tensor = torch.from_numpy(seq).unsqueeze(0)  # (1, seq_len, 18)
        flat = seq[-1:, :]                                # (1, 18)

        proba = self._ensemble.predict_proba(seq_tensor, flat)  # (1,)
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
        """
        Register UE serving assignments for this tick via set_ue_override().

        FIX: Previously set ue.serving_gnb_id directly, which was overwritten
        by the physics engine on the same tick. Now uses the engine's override
        API so the assignment survives exactly one full tick.

        Actions:
        - 0: no-op
        - 1: soft rebalance — move 1 UE from most-loaded to least-loaded cell
        - 2: aggressive rebalance — move up to 3 UEs from cells > 0.7 to cells < 0.5
        - 3: emergency offload — move ALL UEs from cells > 0.9 to least-loaded cell

        Args:
            action: Integer action index.
        """
        if action == 0:
            return

        loads = [g["load"] for g in self._state.gnb_states]

        if action == 1:
            # Soft rebalance: override 1 UE from most-loaded -> least-loaded
            src = int(np.argmax(loads))
            dst = int(np.argmin(loads))
            if src == dst:
                return
            candidates = [u for u in self._sim.ues if u.serving_gnb_id == src]
            if candidates:
                self._sim.set_ue_override(candidates[0].ue_id, dst)

        elif action == 2:
            # Aggressive rebalance: override up to 3 UEs from overloaded to underloaded
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
            # Emergency offload: override ALL UEs from critical cells
            dst = int(np.argmin(loads))
            critical = [i for i, l in enumerate(loads)
                        if l > _LOAD_CRITICAL and i != dst]
            for src in critical:
                for ue in self._sim.ues:
                    if ue.serving_gnb_id == src:
                        self._sim.set_ue_override(ue.ue_id, dst)

    def _maybe_inject_congestion(self) -> None:
        """
        Occasionally inject a load spike to create congestion training scenarios.

        FIX Bug 2: The live simulation runs at avg load 0.37 and never reaches
        congestion thresholds naturally. This method forces one UE to the most
        loaded cell with probability congestion_injection_prob each tick.

        This gives the agent meaningful congestion events to learn from without
        permanently distorting the simulation physics.
        """
        if self._state is None:
            return
        if self.np_random.random() >= self._congestion_injection_prob:
            return

        loads = [g["load"] for g in self._state.gnb_states]
        # Target the most loaded cell to amplify the spike
        target_cell = int(np.argmax(loads))

        # Find a UE NOT already on the target cell and override it there
        candidates = [u for u in self._sim.ues if u.serving_gnb_id != target_cell]
        if candidates:
            # Pick a random candidate for variety
            idx = int(self.np_random.integers(0, len(candidates)))
            self._sim.set_ue_override(candidates[idx].ue_id, target_cell)
            logger.debug(
                "Congestion injection: UE %d -> cell %d (load=%.2f)",
                candidates[idx].ue_id, target_cell, loads[target_cell]
            )

    def _compute_reward(
        self,
        prev_loads: list[float],
        new_loads: list[float],
        handovers: int,
    ) -> tuple[float, dict]:
        """
        Compute the shaped reward for this step.

        Args:
            prev_loads: Cell loads before the action [l0, l1, l2].
            new_loads: Cell loads after the action and tick [l0, l1, l2].
            handovers: Number of handovers triggered this tick.

        Returns:
            Tuple of (clipped reward float, breakdown dict).
        """
        r_drop = 0.0
        r_healthy = 0.0
        r_critical = 0.0
        r_warning = 0.0

        for prev, new in zip(prev_loads, new_loads):
            if new > _LOAD_CRITICAL:
                r_critical += _R_CRITICAL
            elif new > _LOAD_WARNING:
                r_warning += _R_WARNING
            elif new >= _LOAD_HEALTHY_LOW:
                r_healthy += _R_HEALTHY
                if prev > _LOAD_WARNING:
                    r_drop += _R_DROP_BELOW_WARNING
            else:
                r_healthy += _R_HEALTHY * 0.5

        r_handover = _R_HANDOVER * handovers
        r_allclear = _R_ALL_CLEAR if all(l < _LOAD_WARNING for l in new_loads) else 0.0

        total = r_drop + r_healthy + r_critical + r_warning + r_handover + r_allclear
        total = float(np.clip(total, -10.0, 10.0))

        breakdown = {
            "drop_bonus": r_drop,
            "healthy": r_healthy,
            "critical_penalty": r_critical,
            "warning_penalty": r_warning,
            "handover_penalty": r_handover,
            "all_clear_bonus": r_allclear,
            "total": total,
        }
        return total, breakdown