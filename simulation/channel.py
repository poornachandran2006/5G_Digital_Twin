"""
simulation/channel.py

Physics engine for the 5G Digital Twin — radio channel modelling.

This module is intentionally stateless: it exposes only pure functions
that operate on NumPy arrays.  All computations are fully vectorised;
there are zero Python for-loops over UEs or gNBs.
"""

# Run from project root: python -m simulation.engine
from __future__ import annotations

import numpy as np

import config

# Speed of light (m/s) — used in free-space path-loss formula
_C_MPS: float = 3.0e8


def compute_path_loss(
    distances: np.ndarray,
    frequency_ghz: float,
    path_loss_exponent: float,
) -> np.ndarray:
    """
    Compute combined free-space + urban path loss using the 3GPP UMa model.

    Formula::

        PL = 20·log10(4·π·d·f / c) + 10·n·log10(d)

    where ``c = 3×10⁸ m/s``, ``d`` is distance in metres, ``f`` is
    carrier frequency in Hz, and ``n`` is the path-loss exponent.

    Distances are clipped to a minimum of 1.0 m before the logarithm is
    applied to avoid ``log(0)`` singularities.

    Args:
        distances: Distance values in metres, shape ``(N,)`` or broadcast-
                   compatible shape.
        frequency_ghz: Carrier frequency in GHz.
        path_loss_exponent: Urban path-loss exponent ``n`` (dimensionless).

    Returns:
        np.ndarray: Path loss in dB, same shape as ``distances``.
    """
    freq_hz: float = frequency_ghz * 1.0e9
    d = np.maximum(distances, 1.0)

    # Reference path loss at 1m (frequency-dependent intercept)
    pl_intercept = 20.0 * np.log10(4.0 * np.pi * freq_hz / _C_MPS)

    # Distance-dependent loss using configured path loss exponent
    pl_distance = 10.0 * path_loss_exponent * np.log10(d)

    return pl_intercept + pl_distance


def compute_sinr(
    ue_positions: np.ndarray,
    gnb_positions: np.ndarray,
    gnb_tx_powers_dbm: np.ndarray,
    gnb_antenna_gains_db: np.ndarray,
    noise_power_dbm: float,
    frequency_ghz: float,
    path_loss_exponent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute SINR for all UEs against all gNBs simultaneously.

    Fully vectorised via NumPy broadcasting — no Python loops.

    Steps:

    1. **Distance matrix** ``(num_ue, num_gnb)`` computed with broadcasting.
    2. **Path-loss matrix** ``(num_ue, num_gnb)`` via :func:`compute_path_loss`.
    3. **Received-power matrix** ``P_rx = P_tx + G_tx − PL``
       (all in dBm / dB), shape ``(num_ue, num_gnb)``.
    4. **Serving gNB** per UE = ``argmax`` of received power over gNB axis.
    5. **SINR** = serving power − 10·log10(Σ interference_linear + noise_linear)
       where interference is the sum of linear received-power from all
       non-serving gNBs.

    Args:
        ue_positions: UE coordinates in metres, shape ``(num_ue, 2)``.
        gnb_positions: gNB coordinates in metres, shape ``(num_gnb, 2)``.
        gnb_tx_powers_dbm: Transmit power per gNB in dBm, shape ``(num_gnb,)``.
        gnb_antenna_gains_db: Antenna gain per gNB in dB, shape ``(num_gnb,)``.
        noise_power_dbm: Thermal noise floor in dBm (scalar).
        frequency_ghz: Carrier frequency in GHz (scalar).
        path_loss_exponent: Urban path-loss exponent (scalar).

    Returns:
        tuple:
            - **sinr_db** ``(num_ue,)`` — SINR in dB for each UE.
            - **serving_gnb_ids** ``(num_ue,)`` int — index of the gNB with
              the highest received power for each UE.
    """
    # --- Step 1: Distance matrix (num_ue, num_gnb) ---
    # ue_positions[:, np.newaxis, :] → (num_ue, 1, 2)
    # gnb_positions[np.newaxis, :, :] → (1, num_gnb, 2)
    delta = ue_positions[:, np.newaxis, :] - gnb_positions[np.newaxis, :, :]  # (num_ue, num_gnb, 2)
    distances = np.linalg.norm(delta, axis=2)  # (num_ue, num_gnb)

    # --- Step 2: Path-loss matrix (num_ue, num_gnb) ---
    path_loss = compute_path_loss(distances, frequency_ghz, path_loss_exponent)

    # --- Step 3: Received power matrix (num_ue, num_gnb) in dBm ---
    # Broadcast gNB-specific power/gain across all UEs
    tx_power = gnb_tx_powers_dbm[np.newaxis, :]        # (1, num_gnb)
    ant_gain = gnb_antenna_gains_db[np.newaxis, :]      # (1, num_gnb)
    rx_power_dbm = tx_power + ant_gain - path_loss      # (num_ue, num_gnb)

    # --- Step 4: Serving gNB = argmax received power ---
    serving_gnb_ids = np.argmax(rx_power_dbm, axis=1)  # (num_ue,)

    # --- Step 5: SINR ---
    num_ue = ue_positions.shape[0]
    num_gnb = gnb_positions.shape[0]

    # Convert all received powers to linear (mW) for summation
    rx_power_linear = np.power(10.0, rx_power_dbm / 10.0)  # (num_ue, num_gnb)

    # Build a mask: True for serving gNB, False for interferers
    gnb_idx = np.arange(num_gnb)[np.newaxis, :]             # (1, num_gnb)
    serving_mask = gnb_idx == serving_gnb_ids[:, np.newaxis]  # (num_ue, num_gnb)

    # Serving power (linear) — shape (num_ue,)
    serving_power_linear = np.sum(rx_power_linear * serving_mask, axis=1)

    # Interference power (linear) — sum of all non-serving gNBs
    interference_linear = np.sum(rx_power_linear * (~serving_mask), axis=1)  # (num_ue,)

    # Noise in linear (mW)
    noise_linear = 10.0 ** (noise_power_dbm / 10.0)

    # SINR in linear, then convert to dB
    sinr_linear = serving_power_linear / (interference_linear + noise_linear)
    sinr_db = 10.0 * np.log10(sinr_linear)  # (num_ue,)

    return sinr_db, serving_gnb_ids.astype(np.int64)


def compute_throughput(
    sinr_db: np.ndarray,
    bandwidth_mhz: float = 20.0,
) -> np.ndarray:
    """
    Estimate achievable throughput using the Shannon-Hartley theorem.

    Formula::

        C = B · log₂(1 + SINR_linear)   [Mbps]

    SINR_dB values are first clipped to ``[SINR_MIN_DB, SINR_MAX_DB]``
    (from ``config.py``) to reflect realistic radio link limits.

    Args:
        sinr_db: SINR values in dB, shape ``(N,)``.
        bandwidth_mhz: Channel bandwidth in MHz (default 20 MHz).

    Returns:
        np.ndarray: Achievable throughput in Mbps, shape ``(N,)``.
    """
    sinr_clipped = np.clip(sinr_db, config.SINR_MIN_DB, config.SINR_MAX_DB)
    sinr_linear = np.power(10.0, sinr_clipped / 10.0)
    throughput_mbps = bandwidth_mhz * np.log2(1.0 + sinr_linear)
    return throughput_mbps


def compute_prb_demand(
    throughput_mbps: np.ndarray,
    max_prb: int,
    total_capacity_mbps: float,
) -> np.ndarray:
    """
    Estimate the number of PRBs required per UE, proportional to throughput.

    Each PRB carries ``capacity_per_prb = total_capacity_mbps / max_prb``
    Mbps.  The demand is rounded up (ceiling) and clamped to ``[1, max_prb]``.

    Args:
        throughput_mbps: Per-UE throughput in Mbps, shape ``(N,)``.
        max_prb: Maximum PRBs available in one cell (from config).
        total_capacity_mbps: Total cell capacity in Mbps (used to derive
            per-PRB capacity).

    Returns:
        np.ndarray: Integer PRB demand per UE, shape ``(N,)``, dtype int.
    """
    capacity_per_prb = total_capacity_mbps / max_prb  # Mbps per PRB
    prb_needed = np.ceil(throughput_mbps / capacity_per_prb).astype(int)
    return np.clip(prb_needed, 1, max_prb)
