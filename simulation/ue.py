"""
simulation/ue.py

UE (User Equipment) — mobile terminal model for the 5G Digital Twin.
Implements Random Waypoint boundary reflection for mobility.
"""

# Run from project root: python -m simulation.engine
from __future__ import annotations

import numpy as np

import config


class UE:
    """
    Represents a single User Equipment (mobile device) in the simulation.

    Mobility model: Random Waypoint with boundary reflection — when a UE
    hits a grid edge its velocity component perpendicular to that edge is
    negated, keeping it within the simulation area.

    Uses ``__slots__`` for memory efficiency.
    """

    __slots__ = (
        "ue_id",
        "position",
        "velocity",
        "demand_mbps",
        "serving_gnb_id",
        "sinr_db",
        "throughput_mbps",
        "is_handover",
        "_rng",
    )

    def __init__(
        self,
        ue_id: int,
        position: np.ndarray,
        velocity: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Initialise a UE.

        Args:
            ue_id: Unique integer identifier.
            position: Initial (x, y) coordinate in metres, shape ``(2,)``.
            velocity: Initial (vx, vy) in m/s, shape ``(2,)``.
            rng: Optional ``numpy.random.Generator`` for reproducibility.
                 If ``None`` a default generator is created.
        """
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

        self.ue_id: int = ue_id
        self.position: np.ndarray = np.array(position, dtype=np.float64)
        self.velocity: np.ndarray = np.array(velocity, dtype=np.float64)

        # Random initial data demand drawn from config bounds
        self.demand_mbps: float = float(
            self._rng.uniform(config.UE_MIN_DEMAND_MBPS, config.UE_MAX_DEMAND_MBPS)
        )

        self.serving_gnb_id: int | None = None
        self.sinr_db: float = 0.0
        self.throughput_mbps: float = 0.0
        self.is_handover: bool = False

    # ------------------------------------------------------------------
    # Mobility
    # ------------------------------------------------------------------

    def update_position(self, grid_size: float) -> None:
        """
        Advance the UE by one tick using its current velocity.

        Implements boundary reflection: if the new position falls outside
        ``[0, grid_size]`` in either dimension the UE is clamped to the
        boundary and the corresponding velocity component is negated
        (elastic / specular reflection).

        Args:
            grid_size: Side length of the square simulation grid in metres.
        """
        self.position += self.velocity

        # X-axis reflection
        if self.position[0] < 0.0:
            self.position[0] = -self.position[0]
            self.velocity[0] = -self.velocity[0]
        elif self.position[0] > grid_size:
            self.position[0] = 2.0 * grid_size - self.position[0]
            self.velocity[0] = -self.velocity[0]

        # Y-axis reflection
        if self.position[1] < 0.0:
            self.position[1] = -self.position[1]
            self.velocity[1] = -self.velocity[1]
        elif self.position[1] > grid_size:
            self.position[1] = 2.0 * grid_size - self.position[1]
            self.velocity[1] = -self.velocity[1]

        # Safety clamp (handles corner cases with large velocities)
        self.position = np.clip(self.position, 0.0, grid_size)

    def change_direction(self) -> None:
        """
        Randomly reassign velocity to simulate arrival at a waypoint.

        The new speed is drawn uniformly from ``[0, UE_MAX_SPEED_MPS]`` and
        the direction is a uniformly random angle in ``[0, 2π)``.
        """
        speed = float(self._rng.uniform(0.0, config.UE_MAX_SPEED_MPS))
        angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
        self.velocity = np.array(
            [speed * np.cos(angle), speed * np.sin(angle)], dtype=np.float64
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Produce a JSON-serialisable snapshot of the current UE state.

        Suitable for transmission over WebSocket to the dashboard.

        Returns:
            dict: State dictionary with keys ``ue_id``, ``position``,
            ``velocity``, ``demand_mbps``, ``serving_gnb_id``,
            ``sinr_db``, ``throughput_mbps``, and ``is_handover``.
        """
        return {
            "ue_id": self.ue_id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "demand_mbps": self.demand_mbps,
            "serving_gnb_id": self.serving_gnb_id,
            "sinr_db": self.sinr_db,
            "throughput_mbps": self.throughput_mbps,
            "is_handover": self.is_handover,
        }
