"""
kpi/calculator.py

KPI computation engine for the 5G Network Digital Twin.

Pure computation class — no I/O, no side effects.
All per-cell aggregations use NumPy; no Python sum() over lists.

Run from project root: python -m kpi.calculator
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from simulation.engine import SimulationState


# ---------------------------------------------------------------------------
# KPI Snapshot dataclass
# ---------------------------------------------------------------------------

@dataclass
class KPISnapshot:
    """
    All KPIs derived from a single :class:`~simulation.engine.SimulationState`.

    Per-cell metrics are lists of length ``NUM_GNB`` (3 cells).
    System-wide metrics are scalars averaged / summed across all UEs.

    ``is_congested`` and ``congestion_level`` carry their default values here;
    :class:`DataGenerator` overrides them during congestion injection intervals.

    Latency model
    -------------
    ``system_avg_latency_ms`` is an approximation based on the inverse of
    per-UE throughput::

        latency_ms_per_ue = 1000 / max(throughput_mbps, 0.1)

    This is a simplified radio-layer proxy — not a full queuing model.

    Packet loss model
    -----------------
    A UE is considered to have packet loss when its SINR falls below
    ``SINR_MIN_DB + 3.0 dB`` — the minimum margin for a stable link.
    ``packet_loss_rate`` is the fraction of such UEs out of all UEs.
    """

    tick: int
    timestamp: float

    # Per-cell metrics (length NUM_GNB = 3)
    cell_loads: list[float]
    cell_throughputs_mbps: list[float]
    cell_ue_counts: list[int]
    cell_avg_sinr_db: list[float]

    # System-wide metrics
    system_throughput_mbps: float
    system_avg_sinr_db: float
    system_avg_latency_ms: float
    handover_count: int
    handover_rate: float
    packet_loss_rate: float

    # Congestion labels — set/overridden by DataGenerator
    is_congested: bool = False
    congestion_level: float = 0.0


# ---------------------------------------------------------------------------
# KPI Calculator
# ---------------------------------------------------------------------------

class KPICalculator:
    """
    Stateless KPI computation class.

    Derives :class:`KPISnapshot` objects from
    :class:`~simulation.engine.SimulationState` using purely vectorised
    NumPy operations.  No file I/O and no print statements.
    """

    def __init__(self) -> None:
        """No external dependencies — pure computation object."""
        self._sinr_loss_threshold: float = config.SINR_MIN_DB + 3.0

    def compute(self, state: SimulationState) -> KPISnapshot:
        """
        Derive all KPIs from a single :class:`~simulation.engine.SimulationState`.

        Computes per-cell and system-wide metrics in one vectorised pass.

        Per-cell aggregation strategy
        ------------------------------
        UE records are grouped by ``serving_gnb_id`` via masked NumPy arrays.
        All reductions (mean, sum, count) are NumPy calls — no Python loops
        over UEs or gNBs are used for arithmetic.

        Args:
            state: A :class:`~simulation.engine.SimulationState` produced by
                   the simulation engine.

        Returns:
            :class:`KPISnapshot`: Populated KPI snapshot for this tick.
        """
        num_gnb: int = len(state.gnb_states)
        num_ue: int = len(state.ue_states)

        # --- Vectorise UE data ---
        sinr_arr = np.array([u["sinr_db"] for u in state.ue_states], dtype=np.float64)
        tput_arr = np.array([u["throughput_mbps"] for u in state.ue_states], dtype=np.float64)
        gnb_ids  = np.array([u["serving_gnb_id"] for u in state.ue_states], dtype=np.int64)

        # --- Per-cell aggregations (NumPy masked operations) ---
        cell_loads: list[float] = []
        cell_throughputs: list[float] = []
        cell_ue_counts: list[int] = []
        cell_avg_sinr: list[float] = []

        for gid in range(num_gnb):
            mask = gnb_ids == gid              # boolean mask — pure NumPy
            count = int(np.sum(mask))

            cell_loads.append(float(state.gnb_states[gid]["load"]))
            cell_throughputs.append(float(np.sum(tput_arr[mask])) if count > 0 else 0.0)
            cell_ue_counts.append(count)
            cell_avg_sinr.append(float(np.mean(sinr_arr[mask])) if count > 0 else 0.0)

        # --- System-wide metrics ---
        system_throughput = float(np.sum(tput_arr))
        system_avg_sinr   = float(np.mean(sinr_arr))

        # Latency: 1000 / max(tput, 0.1) per UE, then averaged
        latency_per_ue   = 1000.0 / np.maximum(tput_arr, 0.1)   # vectorised
        system_avg_latency = float(np.mean(latency_per_ue))

        # Packet loss: fraction of UEs with SINR below stable-link threshold
        below_threshold  = sinr_arr < self._sinr_loss_threshold
        packet_loss_rate = float(np.sum(below_threshold)) / num_ue

        handover_rate = state.handover_count / num_ue

        # Congestion labels (defaults; DataGenerator overrides during injection)
        max_load = float(np.max(np.array(cell_loads)))
        is_congested    = max_load > config.CELL_LOAD_CRITICAL
        congestion_level = max_load

        return KPISnapshot(
            tick=state.tick,
            timestamp=state.timestamp,
            cell_loads=cell_loads,
            cell_throughputs_mbps=cell_throughputs,
            cell_ue_counts=cell_ue_counts,
            cell_avg_sinr_db=cell_avg_sinr,
            system_throughput_mbps=system_throughput,
            system_avg_sinr_db=system_avg_sinr,
            system_avg_latency_ms=system_avg_latency,
            handover_count=state.handover_count,
            handover_rate=handover_rate,
            packet_loss_rate=packet_loss_rate,
            is_congested=is_congested,
            congestion_level=congestion_level,
        )
