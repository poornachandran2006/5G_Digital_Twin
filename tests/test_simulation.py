"""
tests/test_simulation.py

Unit tests for Phase 2 — 5G Network Digital Twin simulation engine.

Covers channel physics, UE mobility, gNB resource management, and
end-to-end engine execution across 7 test cases.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

# Ensure project root is on path for all imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from simulation import channel
from simulation.gnb import GNB
from simulation.ue import UE
from simulation.engine import NetworkSimulation, SimulationState


# ---------------------------------------------------------------------------
# Helper fixtures / data
# ---------------------------------------------------------------------------

_NUM_UE = 20
_NUM_GNB = 3

# Three gNBs matching the engine layout
_GNB_POSITIONS = np.array([[200.0, 500.0], [500.0, 200.0], [800.0, 700.0]])
_GNB_TX_POWERS = np.full(_NUM_GNB, config.GNB_TX_POWER_DBM)
_GNB_ANTENNA_GAINS = np.full(_NUM_GNB, config.GNB_ANTENNA_GAIN_DB)

# 20 UEs at random but reproducible positions
_rng = np.random.default_rng(42)
_UE_POSITIONS = _rng.uniform(0.0, config.GRID_SIZE_M, size=(_NUM_UE, 2))


# ---------------------------------------------------------------------------
# Test 1 — Path loss increases monotonically with distance
# ---------------------------------------------------------------------------

def test_path_loss_increases_with_distance() -> None:
    """
    Path loss at 100 m must be strictly less than path loss at 500 m.

    Validates that :func:`~simulation.channel.compute_path_loss` is
    monotonically increasing with distance, as required by the 3GPP UMa model.
    """
    pl_100 = channel.compute_path_loss(
        np.array([100.0]),
        config.GNB_FREQUENCY_GHZ,
        config.PATH_LOSS_EXPONENT,
    )
    pl_500 = channel.compute_path_loss(
        np.array([500.0]),
        config.GNB_FREQUENCY_GHZ,
        config.PATH_LOSS_EXPONENT,
    )
    assert pl_100[0] < pl_500[0], (
        f"Expected PL(100m) < PL(500m), got {pl_100[0]:.2f} vs {pl_500[0]:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 2 — SINR output shape is (num_ue,)
# ---------------------------------------------------------------------------

def test_sinr_shape() -> None:
    """
    :func:`~simulation.channel.compute_sinr` must return arrays of shape
    ``(num_ue,)`` for 20 UEs and 3 gNBs.
    """
    sinr_db, serving_ids = channel.compute_sinr(
        ue_positions=_UE_POSITIONS,
        gnb_positions=_GNB_POSITIONS,
        gnb_tx_powers_dbm=_GNB_TX_POWERS,
        gnb_antenna_gains_db=_GNB_ANTENNA_GAINS,
        noise_power_dbm=config.NOISE_POWER_DBM,
        frequency_ghz=config.GNB_FREQUENCY_GHZ,
        path_loss_exponent=config.PATH_LOSS_EXPONENT,
    )
    assert sinr_db.shape == (_NUM_UE,), f"Expected ({_NUM_UE},), got {sinr_db.shape}"
    assert serving_ids.shape == (_NUM_UE,), (
        f"Expected ({_NUM_UE},), got {serving_ids.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3 — UE very close to gNB-0 is served by gNB-0
# ---------------------------------------------------------------------------

def test_sinr_nearest_gnb_has_highest_power() -> None:
    """
    A UE placed 1 m from gNB-0 must be served by gNB-0.

    Validates that :func:`~simulation.channel.compute_sinr` correctly
    selects the nearest (highest received-power) gNB as the serving cell.
    """
    # Place a single UE 1 m north of gNB-0
    ue_near_gnb0 = np.array([[_GNB_POSITIONS[0, 0], _GNB_POSITIONS[0, 1] + 1.0]])

    _, serving_ids = channel.compute_sinr(
        ue_positions=ue_near_gnb0,
        gnb_positions=_GNB_POSITIONS,
        gnb_tx_powers_dbm=_GNB_TX_POWERS,
        gnb_antenna_gains_db=_GNB_ANTENNA_GAINS,
        noise_power_dbm=config.NOISE_POWER_DBM,
        frequency_ghz=config.GNB_FREQUENCY_GHZ,
        path_loss_exponent=config.PATH_LOSS_EXPONENT,
    )
    assert serving_ids[0] == 0, (
        f"UE adjacent to gNB-0 should be served by gNB-0, got gNB-{serving_ids[0]}"
    )


# ---------------------------------------------------------------------------
# Test 4 — All throughput values are positive
# ---------------------------------------------------------------------------

def test_throughput_positive() -> None:
    """
    :func:`~simulation.channel.compute_throughput` must return strictly
    positive values for all UE SINR values after clipping.
    """
    sinr_db, _ = channel.compute_sinr(
        ue_positions=_UE_POSITIONS,
        gnb_positions=_GNB_POSITIONS,
        gnb_tx_powers_dbm=_GNB_TX_POWERS,
        gnb_antenna_gains_db=_GNB_ANTENNA_GAINS,
        noise_power_dbm=config.NOISE_POWER_DBM,
        frequency_ghz=config.GNB_FREQUENCY_GHZ,
        path_loss_exponent=config.PATH_LOSS_EXPONENT,
    )
    throughput = channel.compute_throughput(sinr_db)
    assert np.all(throughput > 0.0), (
        f"Expected all throughputs > 0; min was {throughput.min():.4f} Mbps"
    )


# ---------------------------------------------------------------------------
# Test 5 — UE stays within grid after 1000 position updates
# ---------------------------------------------------------------------------

def test_ue_stays_in_grid() -> None:
    """
    After 1 000 calls to :meth:`~simulation.ue.UE.update_position`, all UE
    positions must remain within ``[0, GRID_SIZE_M]`` in both dimensions.

    Validates the boundary reflection (elastic bounce) logic.
    """
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, config.GRID_SIZE_M, size=2)
    vel = np.array([config.UE_MAX_SPEED_MPS, config.UE_MAX_SPEED_MPS])
    ue = UE(ue_id=0, position=pos, velocity=vel, rng=rng)

    for _ in range(1000):
        ue.update_position(config.GRID_SIZE_M)
        assert 0.0 <= ue.position[0] <= config.GRID_SIZE_M, (
            f"x={ue.position[0]:.2f} out of bounds [0, {config.GRID_SIZE_M}]"
        )
        assert 0.0 <= ue.position[1] <= config.GRID_SIZE_M, (
            f"y={ue.position[1]:.2f} out of bounds [0, {config.GRID_SIZE_M}]"
        )


# ---------------------------------------------------------------------------
# Test 6 — gNB load is always in [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_gnb_load_within_bounds() -> None:
    """
    :meth:`~simulation.gnb.GNB.get_load` must always return a value in
    ``[0.0, 1.0]`` regardless of how many PRBs are allocated.

    Tests the zero-allocation state and a fully-loaded cell.
    """
    gnb = GNB(gnb_id=0, position=(500.0, 500.0))

    # Empty cell
    assert 0.0 <= gnb.get_load() <= 1.0, f"Load at 0 PRBs: {gnb.get_load()}"

    # Fill to capacity
    gnb.allocate_prbs(config.GNB_MAX_PRB)
    assert 0.0 <= gnb.get_load() <= 1.0, (
        f"Load at max PRBs: {gnb.get_load()}"
    )

    # Attempting to over-allocate must be rejected
    overflow_accepted = gnb.allocate_prbs(1)
    assert not overflow_accepted, "allocate_prbs should return False when cell is full"
    assert gnb.get_load() == 1.0, (
        f"Load should remain 1.0 after rejected allocation, got {gnb.get_load()}"
    )


# ---------------------------------------------------------------------------
# Test 7 — Simulation runs 100 ticks without error
# ---------------------------------------------------------------------------

def test_simulation_runs_100_ticks() -> None:
    """
    :meth:`~simulation.engine.NetworkSimulation.run` must complete
    100 ticks without raising any exception and yield exactly 100
    :class:`~simulation.engine.SimulationState` objects.
    """
    sim = NetworkSimulation()
    states: list[SimulationState] = list(sim.run(ticks=100))

    assert len(states) == 100, (
        f"Expected 100 SimulationState objects, got {len(states)}"
    )
    for i, state in enumerate(states):
        assert isinstance(state, SimulationState), (
            f"Tick {i}: expected SimulationState, got {type(state)}"
        )
        assert state.tick == i, f"Tick index mismatch: expected {i}, got {state.tick}"
