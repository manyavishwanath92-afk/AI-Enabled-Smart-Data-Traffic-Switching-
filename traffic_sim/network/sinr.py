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

TWO INTERFERENCE MODELS are available (full rationale in
config.config's module docstring):

  LEGACY_LINEAR_MW  I = coeff * active_users            (Phase 1 baseline)
  NOISE_RISE        I = coeff * active_users * noise    (Phase 2 default)

`compute_sinr`'s `model` argument defaults to LEGACY_LINEAR_MW purely for
backwards compatibility, so Phase 1 callers and unit tests keep their
exact original behaviour. The simulator does NOT rely on that default —
it passes `config.interference_model` explicitly, and the config default
is "noise_rise".
"""

from __future__ import annotations

import math

LEGACY_LINEAR_MW = "legacy_linear_mw"
NOISE_RISE = "noise_rise"


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


def interference_noise_rise_linear(active_users: int, interference_rise_coeff: float, noise_linear: float) -> float:
    """I_linear = interference_rise_coeff * active_users * noise_linear.

    Interference-over-thermal ("noise rise"): interference is expressed
    as a multiple of the noise floor rather than as an absolute milliwatt
    figure. That keeps it dimensionally consistent with the signal and
    noise terms it is summed against, which the legacy model is not — see
    config.config's module docstring for the numbers.
    """
    return interference_rise_coeff * active_users * noise_linear


def sinr_threshold_linear(threshold_db: float) -> float:
    """Convert the SINR service-continuity threshold from dB to linear."""
    return 10 ** (threshold_db / 10)


def compute_sinr(
    rx_power_dbm: float,
    active_users: int,
    noise_floor_dbm: float,
    interference_coeff: float,
    model: str = LEGACY_LINEAR_MW,
) -> float:
    """SINR = Signal_linear / (Interference_linear + Noise_linear).

    All three terms are converted to / expressed in linear units before
    the ratio is formed; no dBm arithmetic occurs here.

    `model` selects how the interference term is computed and defaults to
    LEGACY_LINEAR_MW so existing Phase 1 callers are unaffected. For
    NOISE_RISE, `interference_coeff` carries
    config.interference_rise_coeff (the caller picks the matching
    coefficient for the model it asks for).
    """
    signal_linear = dbm_to_linear_mw(rx_power_dbm)
    noise_linear = dbm_to_linear_mw(noise_floor_dbm)

    if model == NOISE_RISE:
        interference = interference_noise_rise_linear(active_users, interference_coeff, noise_linear)
    elif model == LEGACY_LINEAR_MW:
        interference = interference_linear(active_users, interference_coeff)
    else:
        raise ValueError(f"unknown interference model: {model!r}")

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
