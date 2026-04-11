"""
simulation/engine.py

Core SimPy-based orchestration engine for the 5G Network Digital Twin.

Ties together gNBs, UEs, the channel physics module, and the mobility
model into a single runnable simulation. Each tick:

1. Clear any single-tick RL overrides from the previous tick.
2. UEs move (mobility model).
3. SINR is computed for every UE simultaneously (vectorised channel).
4. UE state is updated with new SINR / serving gNB / throughput.
   RL overrides take precedence over physics-computed serving cell.
5. gNB PRB allocations are reset and reassigned.
6. Handovers are detected by comparing serving gNB from the previous tick.
7. A SimulationState snapshot is captured and yielded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator

import numpy as np
import simpy

import config
from simulation.gnb import GNB
from simulation.ue import UE
from simulation import channel
from simulation.mobility import RandomWaypointMobility

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation state dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationState:
    """
    Immutable snapshot of the simulation at a single tick.

    Attributes:
        tick: Zero-based tick index.
        timestamp: Simulated time in seconds (tick x TICK_DURATION_S).
        gnb_states: List of per-gNB serialised state dicts.
        ue_states: List of per-UE serialised state dicts.
        handover_count: Number of UEs that changed serving gNB this tick.
        avg_sinr_db: Mean SINR across all UEs in dB.
        avg_throughput_mbps: Mean throughput across all UEs in Mbps.
    """

    tick: int
    timestamp: float
    gnb_states: list[dict] = field(default_factory=list)
    ue_states: list[dict] = field(default_factory=list)
    handover_count: int = 0
    avg_sinr_db: float = 0.0
    avg_throughput_mbps: float = 0.0


# ---------------------------------------------------------------------------
# Network simulation
# ---------------------------------------------------------------------------

class NetworkSimulation:
    """
    SimPy-based 5G network simulation orchestrator.

    Instantiates 3 gNBs, 20 UEs, and a Random Waypoint mobility model.
    Exposes run() as a generator over SimulationState objects.

    RL integration: The agent can call set_ue_override(ue_id, gnb_id) before
    a tick to force a UE to a specific cell for that tick only. Physics resumes
    on the following tick. This is the correct way to apply RL actions without
    fighting the physics engine.
    """

    _GNB_POSITIONS: list[tuple[int, int]] = [
        (200, 500),
        (500, 200),
        (800, 700),
    ]

    def __init__(self) -> None:
        """
        Construct the simulation.

        Creates:
        - A SimPy Environment.
        - NUM_GNB gNB instances at fixed positions.
        - A RandomWaypointMobility model (seed 42) which constructs all NUM_UE UE objects.
        - An empty RL override dict for single-tick UE reassignments.
        """
        self._env: simpy.Environment = simpy.Environment()

        self.gnbs: list[GNB] = [
            GNB(gnb_id=i, position=pos)
            for i, pos in enumerate(self._GNB_POSITIONS)
        ]

        self._mobility: RandomWaypointMobility = RandomWaypointMobility(
            num_ue=config.NUM_UE,
            grid_size=config.GRID_SIZE_M,
            max_speed=config.UE_MAX_SPEED_MPS,
            seed=42,
        )
        self.ues: list[UE] = self._mobility.ues

        self._prev_serving: np.ndarray = np.full(config.NUM_UE, -1, dtype=np.int64)
        self._current_state: SimulationState | None = None

        # RL override dict: ue_id -> gnb_id for the current tick only.
        # Set via set_ue_override(), cleared automatically at the start of each tick.
        self._ue_overrides: dict[int, int] = {}

        self._gnb_positions: np.ndarray = np.array(
            [g.position for g in self.gnbs], dtype=np.float64
        )
        self._gnb_tx_powers: np.ndarray = np.array(
            [g.tx_power_dbm for g in self.gnbs], dtype=np.float64
        )
        self._gnb_antenna_gains: np.ndarray = np.array(
            [g.antenna_gain_db for g in self.gnbs], dtype=np.float64
        )

        self._total_capacity_mbps: float = channel.compute_throughput(
            np.array([config.SINR_MAX_DB])
        )[0]

        logger.debug("NetworkSimulation initialised: %d gNBs, %d UEs", len(self.gnbs), len(self.ues))

    # ------------------------------------------------------------------
    # RL override API
    # ------------------------------------------------------------------

    def set_ue_override(self, ue_id: int, gnb_id: int) -> None:
        """
        Force a UE to a specific serving cell for the next simulation tick.

        The override is applied INSTEAD of the physics-computed best cell
        for exactly one tick. On the following tick, physics resumes normally.

        This is the correct integration point for RL actions — it lets the
        agent influence UE placement without permanently breaking the physics.

        Args:
            ue_id: ID of the UE to override.
            gnb_id: ID of the target gNB cell (0, 1, or 2).
        """
        if 0 <= gnb_id < len(self.gnbs):
            self._ue_overrides[ue_id] = gnb_id
        else:
            logger.warning("set_ue_override: invalid gnb_id=%d, ignoring", gnb_id)

    def clear_overrides(self) -> None:
        """
        Clear all pending RL overrides.

        Called automatically at the start of each tick. Should not need to
        be called manually in normal usage.
        """
        self._ue_overrides.clear()

    # ------------------------------------------------------------------
    # SimPy process
    # ------------------------------------------------------------------

    def run_tick(self, env: simpy.Environment) -> Generator:
        """
        SimPy process representing one simulation tick.

        Orchestrates: clear overrides -> mobility -> channel -> state updates
        (with RL override precedence) -> allocation -> handover detection.

        Args:
            env: The SimPy environment driving this process.

        Yields:
            simpy.events.Timeout of 1 tick duration after each tick.
        """
        tick: int = 0

        while True:
            # --- 0. Clear single-tick RL overrides from previous tick ---
            # Overrides are consumed once per tick. Physics resumes next tick.
            self.clear_overrides()

            # --- 1. Mobility ---
            self._mobility.step(self.ues)

            # --- 2. Channel: gather all UE positions as a matrix ---
            ue_positions: np.ndarray = np.array(
                [ue.position for ue in self.ues], dtype=np.float64
            )

            sinr_db, serving_ids = channel.compute_sinr(
                ue_positions=ue_positions,
                gnb_positions=self._gnb_positions,
                gnb_tx_powers_dbm=self._gnb_tx_powers,
                gnb_antenna_gains_db=self._gnb_antenna_gains,
                noise_power_dbm=config.NOISE_POWER_DBM,
                frequency_ghz=config.GNB_FREQUENCY_GHZ,
                path_loss_exponent=config.PATH_LOSS_EXPONENT,
            )

            throughput_mbps: np.ndarray = channel.compute_throughput(sinr_db)

            demand_mbps: np.ndarray = np.array(
                [ue.demand_mbps for ue in self.ues], dtype=np.float64
            )
            prb_demand: np.ndarray = channel.compute_prb_demand(
                throughput_mbps=demand_mbps,
                max_prb=config.GNB_MAX_PRB,
                total_capacity_mbps=self._total_capacity_mbps,
            )

            # --- 3. Update each UE's state ---
            # RL overrides take precedence over physics-computed serving cell.
            # This is the key fix: the engine respects the agent's decisions
            # for exactly one tick before physics resumes.
            for idx, ue in enumerate(self.ues):
                ue.sinr_db = float(sinr_db[idx])
                ue.throughput_mbps = float(throughput_mbps[idx])

                if ue.ue_id in self._ue_overrides:
                    # Agent override: use the RL-assigned cell this tick
                    ue.serving_gnb_id = self._ue_overrides[ue.ue_id]
                    logger.debug(
                        "Tick %d: UE %d overridden to gNB %d (physics said %d)",
                        tick, ue.ue_id, ue.serving_gnb_id, int(serving_ids[idx])
                    )
                else:
                    # Normal physics: connect to best SINR cell
                    ue.serving_gnb_id = int(serving_ids[idx])

            # --- 4. Reset all gNB allocations ---
            for gnb in self.gnbs:
                gnb.reset_allocation()

            # --- 5. Allocate PRBs to serving gNBs ---
            for idx, ue in enumerate(self.ues):
                gnb = self.gnbs[ue.serving_gnb_id]
                success = gnb.allocate_prbs(int(prb_demand[idx]))
                if success:
                    gnb.connected_ues.append(ue.ue_id)

            # --- 6. Detect handovers ---
            # A handover occurs when a UE changes serving gNB from the previous tick,
            # regardless of whether the change was physics-driven or RL-driven.
            handover_count: int = 0
            current_serving = np.array([ue.serving_gnb_id for ue in self.ues], dtype=np.int64)

            for idx, ue in enumerate(self.ues):
                if self._prev_serving[idx] != -1 and self._prev_serving[idx] != current_serving[idx]:
                    ue.is_handover = True
                    handover_count += 1
                else:
                    ue.is_handover = False

            self._prev_serving = current_serving.copy()

            # --- 7. Collect SimulationState ---
            self._current_state = SimulationState(
                tick=tick,
                timestamp=float(env.now),
                gnb_states=[g.to_dict() for g in self.gnbs],
                ue_states=[u.to_dict() for u in self.ues],
                handover_count=handover_count,
                avg_sinr_db=float(np.mean(sinr_db)),
                avg_throughput_mbps=float(np.mean(throughput_mbps)),
            )

            tick += 1
            yield env.timeout(config.TICK_DURATION_S)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, ticks: int) -> Generator[SimulationState, None, None]:
        """
        Run the simulation for ticks steps and yield each state.

        Args:
            ticks: Number of simulation ticks to execute.

        Yields:
            SimulationState: State snapshot after each tick completes.
        """
        env = simpy.Environment()
        self._env = env

        self._prev_serving = np.full(config.NUM_UE, -1, dtype=np.int64)
        self._ue_overrides.clear()

        proc = env.process(self.run_tick(env))

        for _ in range(ticks):
            env.step()
            if self._current_state is not None:
                yield self._current_state

    def get_state(self) -> SimulationState | None:
        """
        Return the most recently computed SimulationState.

        Returns:
            SimulationState | None: Latest snapshot, or None if the
            simulation has not yet run.
        """
        return self._current_state