"""
simulation/mobility.py

Random Waypoint Mobility Model for the 5G Digital Twin.

All UE positions and velocities are managed here. Every 30 ticks a UE
is assigned a new random waypoint (direction + speed) to simulate the
arrival-at-waypoint event characteristic of the Random Waypoint model.
"""

# Run from project root: python -m simulation.engine
from __future__ import annotations

import numpy as np

import config
from simulation.ue import UE


class RandomWaypointMobility:
    """
    Random Waypoint mobility model with boundary reflection.

    Manages a collection of :class:`~simulation.ue.UE` objects and drives
    their movement each simulation tick.  Every ``WAYPOINT_INTERVAL`` ticks
    each UE independently draws a new random direction and speed.

    Reproducibility is guaranteed through a seeded ``numpy.random.Generator``.
    """

    #: Number of ticks between forced waypoint changes per UE.
    WAYPOINT_INTERVAL: int = 30

    def __init__(
        self,
        num_ue: int,
        grid_size: float,
        max_speed: float,
        seed: int = 42,
    ) -> None:
        """
        Initialise the mobility model and create all UE objects.

        Args:
            num_ue: Total number of UEs to manage.
            grid_size: Side length of the square simulation grid in metres.
            max_speed: Maximum UE speed in m/s (from config).
            seed: Integer seed for the random number generator.  Use the
                  same seed across runs for reproducible simulations.
        """
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._grid_size: float = grid_size
        self._max_speed: float = max_speed
        self._num_ue: int = num_ue

        # Per-UE tick counter — tracks when the next waypoint change occurs
        self._tick_counters: np.ndarray = np.zeros(num_ue, dtype=np.int64)

        # Initialise UEs with random positions and velocities
        self.ues: list[UE] = self._create_ues()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_ues(self) -> list[UE]:
        """
        Create all UE objects with random initial positions and velocities.

        Returns:
            list[UE]: Initialised UE list of length ``num_ue``.
        """
        ues: list[UE] = []
        for i in range(self._num_ue):
            # Each UE gets its own sub-generator derived from the master seed
            # so their random streams remain independent and reproducible.
            ue_seed = int(self._rng.integers(0, 2**31))
            ue_rng = np.random.default_rng(ue_seed)

            # Random position uniformly distributed across the grid
            pos = ue_rng.uniform(0.0, self._grid_size, size=2)

            # Random speed and direction
            speed = float(ue_rng.uniform(0.0, self._max_speed))
            angle = float(ue_rng.uniform(0.0, 2.0 * np.pi))
            vel = np.array([speed * np.cos(angle), speed * np.sin(angle)])

            ues.append(UE(ue_id=i, position=pos, velocity=vel, rng=ue_rng))

        return ues

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, ues: list[UE]) -> None:
        """
        Advance all UEs by one simulation tick.

        For each UE:

        * :meth:`~simulation.ue.UE.update_position` is called to move the
          UE by its current velocity and apply boundary reflection.
        * Every :attr:`WAYPOINT_INTERVAL` ticks, a new direction and speed
          are chosen via :meth:`~simulation.ue.UE.change_direction`.

        Args:
            ues: List of :class:`~simulation.ue.UE` objects to update.
                 Typically ``self.ues``, but accepts any list so the engine
                 can pass its own UE references.
        """
        for idx, ue in enumerate(ues):
            ue.update_position(self._grid_size)

            self._tick_counters[idx] += 1
            if self._tick_counters[idx] >= self.WAYPOINT_INTERVAL:
                ue.change_direction()
                self._tick_counters[idx] = 0
