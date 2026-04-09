"""
kpi/data_generator.py

Data generation pipeline for the 5G Network Digital Twin.

Orchestrates simulation + KPI computation + congestion injection +
batch storage into SQLite.  This produces the labelled dataset used
by Phase 4 (ML prediction).

Run from project root: python -m kpi.data_generator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from simulation.engine import NetworkSimulation, SimulationState
from kpi.calculator import KPICalculator, KPISnapshot
from kpi.storage import KPIStorage


# Default congestion injection schedule (start_tick, end_tick, gnb_id)
_DEFAULT_INJECTION: list[tuple[int, int, int]] = [
    (1000, 1200, 0),   # Cell 0 congested for 200 ticks
    (3000, 3300, 1),   # Cell 1 congested for 300 ticks
    (5500, 5700, 2),   # Cell 2 congested for 200 ticks
    (7000, 7400, 0),   # Cell 0 again for 400 ticks
    (9000, 9200, 1),   # Cell 1 again for 200 ticks
]


class DataGenerator:
    """
    Full data-generation pipeline for the 5G Digital Twin.

    Runs the :class:`~simulation.engine.NetworkSimulation`, computes KPIs
    via :class:`~kpi.calculator.KPICalculator`, injects synthetic congestion
    events, and stores everything in batches via :class:`~kpi.storage.KPIStorage`.

    Reproducibility is guaranteed through a seeded ``numpy.random.Generator``
    (``seed=42`` by default).
    """

    def __init__(self, seed: int = 42) -> None:
        """
        Initialise all pipeline components.

        Args:
            seed: RNG seed for congestion-load sampling.  Does not affect
                  the simulation's own seed (fixed at 42 in
                  :class:`~simulation.engine.NetworkSimulation`).
        """
        self.sim: NetworkSimulation = NetworkSimulation()
        self.kpi_calc: KPICalculator = KPICalculator()
        self.storage: KPIStorage = KPIStorage()
        self.rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_injection_set(
        intervals: list[tuple[int, int, int]],
    ) -> dict[int, tuple[int, ...]]:
        """
        Build a tick → (gnb_ids…) lookup for fast interval membership test.

        Args:
            intervals: List of ``(start, end, gnb_id)`` injection intervals.

        Returns:
            dict mapping each injected tick to a tuple of affected gNB IDs.
        """
        inject: dict[int, list[int]] = {}
        for start, end, gid in intervals:
            for t in range(start, end + 1):
                inject.setdefault(t, []).append(gid)
        return {t: tuple(gids) for t, gids in inject.items()}

    def _inject_congestion(
        self,
        snapshot: KPISnapshot,
        gnb_ids: tuple[int, ...],
    ) -> KPISnapshot:
        """
        Artificially set load on one or more cells to simulate a traffic surge.

        For each ``gnb_id`` in ``gnb_ids``, overwrite ``cell_loads[gnb_id]``
        with a value drawn from ``Uniform(0.91, 0.99)`` and update the
        ``is_congested`` / ``congestion_level`` labels accordingly.

        Args:
            snapshot: The :class:`KPISnapshot` to modify (mutated in place).
            gnb_ids: Tuple of gNB indices to mark as congested.

        Returns:
            The modified :class:`KPISnapshot`.
        """
        for gid in gnb_ids:
            injected_load = float(self.rng.uniform(0.91, 0.99))
            snapshot.cell_loads[gid] = injected_load

        snapshot.congestion_level = float(max(snapshot.cell_loads))
        snapshot.is_congested = True
        return snapshot

    # ------------------------------------------------------------------
    # Primary run loop
    # ------------------------------------------------------------------

    def run(
        self,
        total_ticks: int = 10800,
        batch_size: int = 100,
        congestion_inject_intervals: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """
        Run the simulation, compute and store KPIs in batches.

        Congestion injection
        --------------------
        During intervals specified in ``congestion_inject_intervals``, the
        ``cell_loads`` of the targeted gNB are replaced with a random value
        drawn from ``Uniform(0.91, 0.99)`` and the KPISnapshot labels
        ``is_congested=True`` and ``congestion_level=<injected>`` are set.
        This simulates real congestion caused by external traffic bursts.

        Batch processing
        ----------------
        KPISnapshots are accumulated in a list.  Every ``batch_size`` ticks
        the batch is flushed to SQLite via a single ``executemany`` call.
        Any remaining rows after the loop are flushed in a final batch.

        Progress
        --------
        Printed every 1 000 ticks::

            Tick 1000/10800 | Congested ticks so far: 42 | Batch inserts: 10

        Args:
            total_ticks: Number of simulation ticks to run.
            batch_size: Number of KPISnapshots per SQLite batch insert.
            congestion_inject_intervals: List of ``(start, end, gnb_id)``
                tuples.  Defaults to :data:`_DEFAULT_INJECTION` when ``None``.
        """
        if congestion_inject_intervals is None:
            congestion_inject_intervals = _DEFAULT_INJECTION

        inject_map = self._build_injection_set(congestion_inject_intervals)

        batch: list[KPISnapshot] = []
        congested_so_far: int = 0
        batch_inserts: int = 0

        for state in self.sim.run(ticks=total_ticks):
            snapshot = self.kpi_calc.compute(state)
            tick = snapshot.tick

            # Congestion injection
            if tick in inject_map:
                snapshot = self._inject_congestion(snapshot, inject_map[tick])
                congested_so_far += 1

            batch.append(snapshot)

            # Flush batch
            if len(batch) >= batch_size:
                self.storage.insert_batch(batch)
                batch.clear()
                batch_inserts += 1

            # Progress report
            if (tick + 1) % 1000 == 0:
                print(
                    f"Tick {tick + 1}/{total_ticks} | "
                    f"Congested ticks so far: {congested_so_far} | "
                    f"Batch inserts: {batch_inserts}"
                )

        # Final partial batch
        if batch:
            self.storage.insert_batch(batch)
            batch_inserts += 1

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def generate_and_export(
        self,
        total_ticks: int = 10800,
        batch_size: int = 100,
        congestion_inject_intervals: list[tuple[int, int, int]] | None = None,
        csv_path: str = "data/kpi_dataset.csv",
    ) -> pd.DataFrame:
        """
        Full pipeline: simulate → compute KPIs → store → export CSV.

        Calls :meth:`run` then exports via
        :meth:`~kpi.storage.KPIStorage.export_csv` and retrieves the full
        :class:`pandas.DataFrame` via
        :meth:`~kpi.storage.KPIStorage.get_dataframe`.

        Prints final statistics::

            Total ticks      : 10800
            Congested ticks  : 1100
            Congestion rate  : 10.2%
            Mean sys tput    : 87.34 Mbps
            Mean cell loads  : C0=0.212 C1=0.198 C2=0.204
            CSV              : data/kpi_dataset.csv

        Args:
            total_ticks: Total simulation ticks to run.
            batch_size: SQLite batch insert size.
            congestion_inject_intervals: Congestion schedule (see :meth:`run`).
            csv_path: Output CSV path.

        Returns:
            pd.DataFrame: Full KPI dataset ready for the ML pipeline.
        """
        self.run(
            total_ticks=total_ticks,
            batch_size=batch_size,
            congestion_inject_intervals=congestion_inject_intervals,
        )
        self.storage.export_csv(csv_path)
        df = self.storage.get_dataframe()

        # Final stats
        total = len(df)
        congested = int(df["is_congested"].sum())
        rate = congested / total * 100.0 if total > 0 else 0.0
        mean_tput = df["system_throughput"].mean()
        c0 = df["cell0_load"].mean()
        c1 = df["cell1_load"].mean()
        c2 = df["cell2_load"].mean()

        print(
            f"\n{'='*55}\n"
            f"  Phase 3 — Data Generation Complete\n"
            f"{'='*55}\n"
            f"  Total ticks      : {total:,}\n"
            f"  Congested ticks  : {congested:,}\n"
            f"  Congestion rate  : {rate:.1f}%\n"
            f"  Mean sys tput    : {mean_tput:.2f} Mbps\n"
            f"  Mean cell loads  : C0={c0:.3f} C1={c1:.3f} C2={c2:.3f}\n"
            f"  CSV              : {csv_path}\n"
            f"{'='*55}"
        )
        return df
