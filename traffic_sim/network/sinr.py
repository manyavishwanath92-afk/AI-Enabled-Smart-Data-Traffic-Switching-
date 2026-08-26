"""Interference, noise and SINR (Guide Section 8.4, 8.5, 8.6).

POWER-UNIT CONSISTENCY RULE: SINR is a ratio of *linear* power
quantities. `received_power_dbm` is expressed in dBm (see
`network.received_power`); before it can be combined with interference
or noise it MUST be converted to linear milliwatts via
`dbm_to_linear_mw`. Never perform SINR arithmetic directly on dBm values
— only the path-loss-to-received-power step stays in the log domain.

    SINR = Signal_linear / (Interference_linear + Noise_linear)

INTERFERENCE MODEL — "noise rise" (interference-over-thermal):

    I_linear = interference_rise_coeff * active_users * noise_linear

Interference is expressed as a multiple of the noise floor rather than
as a bare milliwatt figure, which keeps it dimensionally consistent with
the signal and noise terms it is summed against. At the project's
default Tx power / noise floor / path loss this yields SINR spanning
roughly 5-41 dB, falling with both distance and cell load, so the 5 dB
service-continuity threshold actually bites at the cell edge under heavy
load.

`interference_rise_coeff` (see config.config.SimConfig) is an explicitly
configurable PoC assumption, NOT a real-world telecom constant. It is
never hardcoded here — every call site must pass it in explicitly from
config, so changing it in config.py alone is enough to run a different
interference sensitivity experiment.
"""

from __future__ import annotations

import math


def dbm_to_linear_mw(dbm: float) -> float:
    """Convert a power expressed in dBm to linear milliwatts."""
    return 10 ** (dbm / 10)


def interference_noise_rise_linear(active_users: int, interference_rise_coeff: float, noise_linear: float) -> float:
    """I_linear = interference_rise_coeff * active_users * noise_linear.

    Interference-over-thermal ("noise rise"): interference is expressed
    as a multiple of the noise floor rather than as an absolute milliwatt
    figure, which keeps it dimensionally consistent with the signal and
    noise terms it is summed against in compute_sinr.
    """
    return interference_rise_coeff * active_users * noise_linear


def sinr_threshold_linear(threshold_db: float) -> float:
    """Convert the SINR service-continuity threshold from dB to linear."""
    return 10 ** (threshold_db / 10)


def compute_sinr(
    rx_power_dbm: float,
    active_users: int,
    noise_floor_dbm: float,
    interference_rise_coeff: float,
) -> float:
    """SINR = Signal_linear / (Interference_linear + Noise_linear).

    All three terms are converted to / expressed in linear units before
    the ratio is formed; no dBm arithmetic occurs here. Interference uses
    the noise-rise model — see module docstring.
    """
    signal_linear = dbm_to_linear_mw(rx_power_dbm)
    noise_linear = dbm_to_linear_mw(noise_floor_dbm)
    interference = interference_noise_rise_linear(active_users, interference_rise_coeff, noise_linear)

    denominator = interference + noise_linear
    if denominator <= 0:
        return float("inf")
    return signal_linear / denominator


def sinr_to_db(sinr_linear: float) -> float:
    """Linear SINR -> dB, for reporting/KPIs."""
    if sinr_linear <= 0:
        return float("-inf")
    return 10 * math.log10(sinr_linear)


def service_continuity_ratio(sinr_observations: list[float], threshold_linear: float) -> float:
    """ServiceContinuity = count(SINR >= threshold) / total observations."""
    if not sinr_observations:
        return 0.0
    hits = sum(1 for s in sinr_observations if s >= threshold_linear)
    return hits / len(sinr_observations)
