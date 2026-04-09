"""
simulation/gnb.py

gNB (Next Generation Node B) — 5G base station model.
All configuration values sourced from config.py — no hardcoded values.
"""

# Run from project root: python -m simulation.engine
from __future__ import annotations

import numpy as np

import config


class GNB:
    """
    Represents a single 5G gNB (base station) in the Digital Twin simulation.

    Uses ``__slots__`` for memory efficiency when many gNB instances coexist.
    All default parameter values are sourced exclusively from ``config.py``.
    """

    __slots__ = (
        "gnb_id",
        "position",
        "tx_power_dbm",
        "antenna_gain_db",
        "max_prb",
        "frequency_ghz",
        "connected_ues",
        "allocated_prbs",
    )

    def __init__(
        self,
        gnb_id: int,
        position: tuple[float, float],
        tx_power_dbm: float = config.GNB_TX_POWER_DBM,
        antenna_gain_db: float = config.GNB_ANTENNA_GAIN_DB,
        max_prb: int = config.GNB_MAX_PRB,
        frequency_ghz: float = config.GNB_FREQUENCY_GHZ,
    ) -> None:
        """
        Initialise a gNB.

        Args:
            gnb_id: Unique integer identifier for this base station.
            position: (x, y) coordinates in metres within the simulation grid.
            tx_power_dbm: Transmit power in dBm (default from config).
            antenna_gain_db: Antenna gain in dB (default from config).
            max_prb: Maximum Physical Resource Blocks available (default from config).
            frequency_ghz: Operating carrier frequency in GHz (default from config).
        """
        self.gnb_id: int = gnb_id
        self.position: np.ndarray = np.array(position, dtype=np.float64)
        self.tx_power_dbm: float = tx_power_dbm
        self.antenna_gain_db: float = antenna_gain_db
        self.max_prb: int = max_prb
        self.frequency_ghz: float = frequency_ghz
        self.connected_ues: list[int] = []
        self.allocated_prbs: int = 0

    # ------------------------------------------------------------------
    # Load / congestion helpers
    # ------------------------------------------------------------------

    def get_load(self) -> float:
        """
        Return the current cell load as a fraction of maximum PRB capacity.

        Returns:
            float: Ratio ``allocated_prbs / max_prb`` in ``[0.0, 1.0]``.
        """
        return self.allocated_prbs / self.max_prb

    def is_congested(self) -> bool:
        """
        Determine whether this cell is in a congested state.

        A cell is considered congested when its load exceeds the critical
        threshold defined in ``config.CELL_LOAD_CRITICAL``.

        Returns:
            bool: ``True`` if ``get_load() > CELL_LOAD_CRITICAL``.
        """
        return self.get_load() > config.CELL_LOAD_CRITICAL

    # ------------------------------------------------------------------
    # Resource allocation
    # ------------------------------------------------------------------

    def reset_allocation(self) -> None:
        """
        Reset PRB allocation and clear the connected-UE list for a new tick.

        Should be called at the start of each simulation tick before new
        resource assignments are made.
        """
        self.allocated_prbs = 0
        self.connected_ues = []

    def allocate_prbs(self, n_prbs: int) -> bool:
        """
        Attempt to allocate ``n_prbs`` Physical Resource Blocks to this cell.

        The allocation succeeds only if the resulting total does not exceed
        ``max_prb``.

        Args:
            n_prbs: Number of PRBs to allocate for one UE.

        Returns:
            bool: ``True`` if allocation succeeded; ``False`` if capacity
            would be exceeded (allocation is NOT applied in the False case).
        """
        if self.allocated_prbs + n_prbs > self.max_prb:
            return False
        self.allocated_prbs += n_prbs
        return True

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Produce a JSON-serialisable snapshot of the current gNB state.

        Suitable for transmission over WebSocket to the dashboard.

        Returns:
            dict: State dictionary with keys ``gnb_id``, ``position``,
            ``tx_power_dbm``, ``antenna_gain_db``, ``max_prb``,
            ``frequency_ghz``, ``connected_ues``, ``allocated_prbs``,
            ``load``, and ``is_congested``.
        """
        return {
            "gnb_id": self.gnb_id,
            "position": self.position.tolist(),
            "tx_power_dbm": self.tx_power_dbm,
            "antenna_gain_db": self.antenna_gain_db,
            "max_prb": self.max_prb,
            "frequency_ghz": self.frequency_ghz,
            "connected_ues": list(self.connected_ues),
            "allocated_prbs": self.allocated_prbs,
            "load": self.get_load(),
            "is_congested": self.is_congested(),
        }
