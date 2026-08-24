"""Interference, noise and SINR (Guide Section 8.4, 8.5, 8.6).

POWER-UNIT CONSISTENCY RULE: SINR is a ratio of *linear* power
quantities. `received_power_dbm` is expressed in dBm (see
`network.received_power`); before it can be combined with interference
or noise it MUST be converted to linear milliwatts via
`dbm_to_linear_mw`. Never perform SINR arithmetic directly on dBm values
— only the path-loss-to-received-power step stays in the log domain.

    SINR = Signal_linear / (Interference_linear + Noise_linear)

`interference_coeff` (see config.config.SimConfig) is an explicitly
configurable PoC assumption, NOT a real-world telecom constant. It is
never hardcoded here — every call site must pass it in explicitly from
config, so changing it in config.py alone is enough to run a different
interference sensitivity experiment.
"""

from __future__ import annotations


def dbm_to_linear_mw(dbm: float) -> float:
    """Convert a power expressed in dBm to linear milliwatts."""
    return 10 ** (dbm / 10)


def interference_linear(active_users: int, interference_coeff: float) -> float:
    """I_linear = interference_coeff * active_users.

    Deliberately simplified, synthetic, linear-domain assumption
    (documented in config.config.SimConfig.interference_coeff) — not a
    physical interference model, and not something derived from a dBm
    reading, so it is not passed through dbm_to_linear_mw.
    """
    return interference_coeff * active_users


def sinr_threshold_linear(threshold_db: float) -> float:
    """Convert the SINR service-continuity threshold from dB to linear."""
    return 10 ** (threshold_db / 10)


def compute_sinr(
    rx_power_dbm: float,
    active_users: int,
    noise_floor_dbm: float,
    interference_coeff: float,
) -> float:
    """SINR = Signal_linear / (Interference_linear + Noise_linear).

    All three terms are converted to / expressed in linear units before
    the ratio is formed; no dBm arithmetic occurs here.
    """
    signal_linear = dbm_to_linear_mw(rx_power_dbm)
    interference = interference_linear(active_users, interference_coeff)
    noise_linear = dbm_to_linear_mw(noise_floor_dbm)
    denominator = interference + noise_linear
    if denominator <= 0:
        return float("inf")
    return signal_linear / denominator


def service_continuity_ratio(sinr_observations: list[float], threshold_linear: float) -> float:
    """ServiceContinuity = count(SINR >= threshold) / total observations."""
    if not sinr_observations:
        return 0.0
    hits = sum(1 for s in sinr_observations if s >= threshold_linear)
    return hits / len(sinr_observations)
