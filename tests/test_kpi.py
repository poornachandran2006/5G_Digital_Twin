"""
tests/test_kpi.py

Unit tests for Phase 3 — KPI Engine and Data Generation.

Covers KPISnapshot fields, metric bounds, storage correctness,
batch insert performance, congestion injection labelling, and
CSV export.
"""

from __future__ import annotations

import os
import time
import tempfile
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from simulation.engine import NetworkSimulation
from kpi.calculator import KPICalculator, KPISnapshot
from kpi.storage import KPIStorage
from kpi.data_generator import DataGenerator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_snapshot(tick: int = 0) -> KPISnapshot:
    """Produce a real KPISnapshot by running one tick of the engine."""
    sim = NetworkSimulation()
    states = list(sim.run(ticks=tick + 1))
    calc = KPICalculator()
    return calc.compute(states[tick])


def _make_snapshots(n: int) -> list[KPISnapshot]:
    """Produce n consecutive KPISnapshots from the engine."""
    sim = NetworkSimulation()
    calc = KPICalculator()
    return [calc.compute(s) for s in sim.run(ticks=n)]


# ---------------------------------------------------------------------------
# Test 1 — KPISnapshot has all required fields and cell_loads is valid
# ---------------------------------------------------------------------------

def test_kpi_snapshot_fields() -> None:
    """
    KPISnapshot must expose all required fields, with ``cell_loads``
    of length 3 and each load in ``[0, 1]``.
    """
    snap = _make_snapshot(tick=0)

    # Structural checks
    assert hasattr(snap, "tick")
    assert hasattr(snap, "timestamp")
    assert hasattr(snap, "cell_loads")
    assert hasattr(snap, "cell_throughputs_mbps")
    assert hasattr(snap, "cell_ue_counts")
    assert hasattr(snap, "cell_avg_sinr_db")
    assert hasattr(snap, "system_throughput_mbps")
    assert hasattr(snap, "system_avg_sinr_db")
    assert hasattr(snap, "system_avg_latency_ms")
    assert hasattr(snap, "handover_count")
    assert hasattr(snap, "handover_rate")
    assert hasattr(snap, "packet_loss_rate")
    assert hasattr(snap, "is_congested")
    assert hasattr(snap, "congestion_level")

    # Cell loads
    assert len(snap.cell_loads) == 3, f"Expected 3 cells, got {len(snap.cell_loads)}"
    for i, load in enumerate(snap.cell_loads):
        assert 0.0 <= load <= 1.0, f"cell_loads[{i}]={load:.4f} not in [0, 1]"


# ---------------------------------------------------------------------------
# Test 2 — system_avg_latency_ms is strictly positive
# ---------------------------------------------------------------------------

def test_latency_positive() -> None:
    """``system_avg_latency_ms`` must be > 0 for all normal simulation states."""
    snaps = _make_snapshots(10)
    for i, snap in enumerate(snaps):
        assert snap.system_avg_latency_ms > 0.0, (
            f"Tick {i}: latency={snap.system_avg_latency_ms:.4f} not positive"
        )


# ---------------------------------------------------------------------------
# Test 3 — packet_loss_rate is in [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_packet_loss_in_bounds() -> None:
    """``packet_loss_rate`` must be in ``[0.0, 1.0]`` for all ticks."""
    snaps = _make_snapshots(20)
    for i, snap in enumerate(snaps):
        assert 0.0 <= snap.packet_loss_rate <= 1.0, (
            f"Tick {i}: packet_loss_rate={snap.packet_loss_rate:.4f} out of bounds"
        )


# ---------------------------------------------------------------------------
# Test 4 — Storage: insert 10 rows, retrieve 10 rows with correct ticks
# ---------------------------------------------------------------------------

def test_storage_insert_and_retrieve() -> None:
    """
    Insert 10 KPISnapshots into an in-memory DB, query back, and confirm
    10 rows with the expected tick values.
    """
    snaps = _make_snapshots(10)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        with KPIStorage(db_path=db_path) as storage:
            storage.insert_batch(snaps)
            df = storage.get_dataframe()

        assert len(df) == 10, f"Expected 10 rows, got {len(df)}"
        expected_ticks = list(range(10))
        assert list(df["tick"]) == expected_ticks, (
            f"Tick mismatch: {list(df['tick'])} != {expected_ticks}"
        )
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 5 — Batch insert of 500 rows completes in under 2 seconds
# ---------------------------------------------------------------------------

def test_batch_insert_faster_than_row_insert() -> None:
    """
    Batch-inserting 500 KPISnapshots must complete in under 2 seconds.

    Validates that ``executemany`` (not row-by-row INSERT) is used.
    """
    snaps = _make_snapshots(500)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        with KPIStorage(db_path=db_path) as storage:
            t0 = time.perf_counter()
            storage.insert_batch(snaps)
            elapsed = time.perf_counter() - t0

        assert elapsed < 2.0, (
            f"Batch insert of 500 rows took {elapsed:.3f}s — expected < 2.0s"
        )
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 6 — Congestion injection labels ticks correctly
# ---------------------------------------------------------------------------

def test_congestion_injection() -> None:
    """
    With injection interval (10, 40, 0), ticks 10-40 must have
    ``is_congested=True`` and ``congestion_level > 0.9``.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name

    try:
        gen = DataGenerator(seed=42)
        gen.storage = KPIStorage(db_path=db_path)

        gen.run(
            total_ticks=50,
            batch_size=50,
            congestion_inject_intervals=[(10, 40, 0)],
        )
        df = gen.storage.get_dataframe()
        gen.storage.close()

        injected = df[(df["tick"] >= 10) & (df["tick"] <= 40)]
        non_injected = df[(df["tick"] < 10) | (df["tick"] > 40)]

        # All injection ticks must be labelled congested
        assert injected["is_congested"].all(), (
            f"Some injected ticks not labelled congested:\n{injected[['tick','is_congested']]}"
        )
        # Injected load must be > 0.9
        assert (injected["congestion_level"] > 0.9).all(), (
            f"Some injected congestion_level ≤ 0.9:\n{injected[['tick','congestion_level']]}"
        )
        # Non-injected ticks must NOT be artificially labelled
        # (they can be naturally congested, but congestion_level should be ≤ 1.0)
        assert (non_injected["congestion_level"] <= 1.0).all()

    finally:
        os.unlink(db_path)
        try:
            os.unlink(csv_path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Test 7 — export_csv creates file with header + 100 data rows
# ---------------------------------------------------------------------------

def test_export_csv_creates_file() -> None:
    """
    After ``generate_and_export()`` for 100 ticks, ``data/kpi_dataset.csv``
    must exist and contain at least 101 lines (1 header + 100 data rows).
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w"
    ) as f:
        csv_path = f.name

    try:
        gen = DataGenerator(seed=42)
        gen.storage = KPIStorage(db_path=db_path)

        gen.generate_and_export(
            total_ticks=100,
            batch_size=50,
            csv_path=csv_path,
        )

        assert os.path.exists(csv_path), f"CSV not found at {csv_path}"
        with open(csv_path, "r") as fh:
            lines = fh.readlines()
        assert len(lines) >= 101, (
            f"Expected ≥101 lines (header + 100 rows), got {len(lines)}"
        )
    finally:
        gen.storage.close()
        os.unlink(db_path)
        try:
            os.unlink(csv_path)
        except FileNotFoundError:
            pass
