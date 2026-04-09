"""
kpi/storage.py

Persistent KPI storage for the 5G Network Digital Twin.

Uses SQLite so no external database server is required.
All writes are batched — executemany is called once per batch, with
a single commit.  Individual INSERT statements are never used.

Run from project root: python -m kpi.storage
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

from kpi.calculator import KPISnapshot


# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS kpi_snapshots (
    tick                 INTEGER PRIMARY KEY,
    timestamp            REAL,
    cell0_load           REAL,
    cell1_load           REAL,
    cell2_load           REAL,
    cell0_throughput     REAL,
    cell1_throughput     REAL,
    cell2_throughput     REAL,
    cell0_ue_count       INTEGER,
    cell1_ue_count       INTEGER,
    cell2_ue_count       INTEGER,
    cell0_avg_sinr       REAL,
    cell1_avg_sinr       REAL,
    cell2_avg_sinr       REAL,
    system_throughput    REAL,
    system_avg_sinr      REAL,
    system_avg_latency_ms REAL,
    handover_count       INTEGER,
    handover_rate        REAL,
    packet_loss_rate     REAL,
    is_congested         INTEGER,
    congestion_level     REAL
)
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO kpi_snapshots VALUES (
    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
)
"""


# ---------------------------------------------------------------------------
# Storage class
# ---------------------------------------------------------------------------

class KPIStorage:
    """
    SQLite-backed KPI store for the 5G Digital Twin simulation.

    Connection lifecycle
    --------------------
    The connection is opened in ``__init__`` and kept alive for the lifetime
    of the object.  Call :meth:`close` (or use as a context manager) when
    done.

    Write strategy
    --------------
    All writes go through :meth:`insert_batch` which calls ``executemany``
    once per batch and commits once.  Individual ``INSERT`` statements are
    never issued, keeping I/O overhead proportional to batch size.
    """

    def __init__(self, db_path: str = "data/kpi_data.db") -> None:
        """
        Open (or create) the SQLite database and ensure the table exists.

        Args:
            db_path: Relative or absolute path to the ``.db`` file.
                     Parent directories are created automatically.
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path: str = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")   # better concurrency
        with self._conn:
            self._conn.execute(_CREATE_TABLE_SQL)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_to_tuple(s: KPISnapshot) -> tuple:
        """Convert a :class:`KPISnapshot` to the positional tuple for INSERT."""
        return (
            s.tick,
            s.timestamp,
            s.cell_loads[0],
            s.cell_loads[1],
            s.cell_loads[2],
            s.cell_throughputs_mbps[0],
            s.cell_throughputs_mbps[1],
            s.cell_throughputs_mbps[2],
            s.cell_ue_counts[0],
            s.cell_ue_counts[1],
            s.cell_ue_counts[2],
            s.cell_avg_sinr_db[0],
            s.cell_avg_sinr_db[1],
            s.cell_avg_sinr_db[2],
            s.system_throughput_mbps,
            s.system_avg_sinr_db,
            s.system_avg_latency_ms,
            s.handover_count,
            s.handover_rate,
            s.packet_loss_rate,
            int(s.is_congested),
            s.congestion_level,
        )

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def insert_batch(self, snapshots: list[KPISnapshot]) -> None:
        """
        Batch-insert a list of :class:`KPISnapshot` objects.

        Converts the entire list to tuples first, then calls
        ``executemany`` once and commits once.  This is orders of magnitude
        faster than row-by-row inserts for large batches.

        Args:
            snapshots: Non-empty list of :class:`KPISnapshot` instances.
        """
        if not snapshots:
            return
        rows = [self._snapshot_to_tuple(s) for s in snapshots]
        with self._conn:
            self._conn.executemany(_INSERT_SQL, rows)

    # ------------------------------------------------------------------
    # Public read / export API
    # ------------------------------------------------------------------

    def export_csv(self, output_path: str = "data/kpi_dataset.csv") -> None:
        """
        Export the full ``kpi_snapshots`` table to a CSV file.

        Uses ``pandas.read_sql`` for a single round-trip, then ``to_csv``.
        Prints row count, file size in KB, and number of congested rows.

        Args:
            output_path: Destination CSV path (created if missing).
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_sql("SELECT * FROM kpi_snapshots ORDER BY tick", self._conn)
        df.to_csv(output_path, index=False)

        size_kb = os.path.getsize(output_path) / 1024.0
        congested_count = int(df["is_congested"].sum())
        print(
            f"Exported {len(df):,} rows -> {output_path} "
            f"({size_kb:.1f} KB) | Congested rows: {congested_count:,}"
        )

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the full dataset as a :class:`pandas.DataFrame`.

        Used by the ML pipeline (Phase 4) as its primary data source.

        Returns:
            pd.DataFrame: All rows from ``kpi_snapshots``, ordered by tick.
        """
        return pd.read_sql("SELECT * FROM kpi_snapshots ORDER BY tick", self._conn)

    def close(self) -> None:
        """Close the SQLite connection, flushing any pending WAL data."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "KPIStorage":
        return self

    def __exit__(self, *_) -> None:
        self.close()
