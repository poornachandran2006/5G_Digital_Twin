"""
simulation/engine.py

Core SimPy-based orchestration engine for the 5G Network Digital Twin.

Ties together gNBs, UEs, the channel physics module, and the mobility
model into a single runnable simulation.  Each tick:

1. UEs move (mobility model).
2. SINR is computed for every UE simultaneously (vectorised channel).
3. UE state is updated with new SINR / serving gNB / throughput.
4. gNB PRB allocations are reset and reassigned.
5. Handovers are detected by comparing serving gNB from the previous tick.
6. A ``SimulationState`` snapshot is captured and yielded.
"""

# Run from project root: python -m simulation.engine
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import numpy as np
import simpy

import config
from simulation.gnb import GNB
from simulation.ue import UE
from simulation import channel
from simulation.mobility import RandomWaypointMobility


# ---------------------------------------------------------------------------
# Simulation state dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationState:
    """
    Immutable snapshot of the simulation at a single tick.

    Attributes:
        tick: Zero-based tick index.
        timestamp: Simulated time in seconds (``tick × TICK_DURATION_S``).
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
    Exposes :meth:`run` as a generator over :class:`SimulationState` objects.

    All random state is seeded to ``42`` for reproducibility.
    """

    # Fixed gNB deployment positions — spread for good coverage
    _GNB_POSITIONS: list[tuple[int, int]] = [
        (200, 500),
        (500, 200),
        (800, 700),
    ]

    def __init__(self) -> None:
        """
        Construct the simulation.

        Creates:
        * A SimPy ``Environment``.
        * ``NUM_GNB`` gNB instances at fixed positions.
        * A ``RandomWaypointMobility`` model (seed 42) which constructs
          all ``NUM_UE`` UE objects.
        """
        self._env: simpy.Environment = simpy.Environment()

        # gNBs — positions from the spec; all RF parameters from config.py
        self.gnbs: list[GNB] = [
            GNB(gnb_id=i, position=pos)
            for i, pos in enumerate(self._GNB_POSITIONS)
        ]

        # Mobility model creates and owns UEs
        self._mobility: RandomWaypointMobility = RandomWaypointMobility(
            num_ue=config.NUM_UE,
            grid_size=config.GRID_SIZE_M,
            max_speed=config.UE_MAX_SPEED_MPS,
            seed=42,
        )
        self.ues: list[UE] = self._mobility.ues

        # Track serving gNB from the previous tick for handover detection
        self._prev_serving: np.ndarray = np.full(config.NUM_UE, -1, dtype=np.int64)

        # Current snapshot — updated each tick
        self._current_state: SimulationState | None = None

        # Pre-compute static gNB arrays for repeated channel calls
        self._gnb_positions: np.ndarray = np.array(
            [g.position for g in self.gnbs], dtype=np.float64
        )
        self._gnb_tx_powers: np.ndarray = np.array(
            [g.tx_power_dbm for g in self.gnbs], dtype=np.float64
        )
        self._gnb_antenna_gains: np.ndarray = np.array(
            [g.antenna_gain_db for g in self.gnbs], dtype=np.float64
        )

        # Approximate total cell capacity used for PRB demand estimation
        # Shannon at SINR_MAX with 20 MHz BW ≈ upper bound
        self._total_capacity_mbps: float = channel.compute_throughput(
            np.array([config.SINR_MAX_DB])
        )[0]

    # ------------------------------------------------------------------
    # SimPy process
    # ------------------------------------------------------------------

    def run_tick(self, env: simpy.Environment) -> Generator:
        """
        SimPy process representing one simulation tick.

        Orchestrates mobility → channel → state updates → allocation →
        handover detection and collects a :class:`SimulationState`.

        Args:
            env: The SimPy environment driving this process.

        Yields:
            simpy.events.Timeout of 1 tick duration after each tick.
        """
        tick: int = 0

        while True:
            # --- 1. Mobility ---
            self._mobility.step(self.ues)

            # --- 2. Channel: gather all UE positions as a matrix ---
            ue_positions: np.ndarray = np.array(
                [ue.position for ue in self.ues], dtype=np.float64
            )  # (num_ue, 2)

            sinr_db, serving_ids = channel.compute_sinr(
                ue_positions=ue_positions,
                gnb_positions=self._gnb_positions,
                gnb_tx_powers_dbm=self._gnb_tx_powers,
                gnb_antenna_gains_db=self._gnb_antenna_gains,
                noise_power_dbm=config.NOISE_POWER_DBM,
                frequency_ghz=config.GNB_FREQUENCY_GHZ,
                path_loss_exponent=config.PATH_LOSS_EXPONENT,
            )  # (num_ue,), (num_ue,)

            throughput_mbps: np.ndarray = channel.compute_throughput(sinr_db)

            prb_demand: np.ndarray = channel.compute_prb_demand(
                throughput_mbps=throughput_mbps,
                max_prb=config.GNB_MAX_PRB,
                total_capacity_mbps=self._total_capacity_mbps,
            )

            # --- 3. Update each UE's state vectors ---
            for idx, ue in enumerate(self.ues):
                ue.sinr_db = float(sinr_db[idx])
                ue.serving_gnb_id = int(serving_ids[idx])
                ue.throughput_mbps = float(throughput_mbps[idx])

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
            handover_count: int = 0
            for idx, ue in enumerate(self.ues):
                current_gnb = int(serving_ids[idx])
                if self._prev_serving[idx] != -1 and self._prev_serving[idx] != current_gnb:
                    ue.is_handover = True
                    handover_count += 1
                else:
                    ue.is_handover = False

            self._prev_serving = serving_ids.copy()

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
        Run the simulation for ``ticks`` steps and yield each state.

        Args:
            ticks: Number of simulation ticks to execute.

        Yields:
            SimulationState: State snapshot after each tick completes.
        """
        env = simpy.Environment()
        self._env = env

        # Re-initialise tracking so run() is idempotent
        self._prev_serving = np.full(config.NUM_UE, -1, dtype=np.int64)

        proc = env.process(self.run_tick(env))

        for _ in range(ticks):
            # Advance simulation by one tick
            env.step()
            if self._current_state is not None:
                yield self._current_state

    def get_state(self) -> SimulationState | None:
        """
        Return the most recently computed :class:`SimulationState`.

        Returns:
            SimulationState | None: Latest snapshot, or ``None`` if the
            simulation has not yet run.
        """
        return self._current_state
