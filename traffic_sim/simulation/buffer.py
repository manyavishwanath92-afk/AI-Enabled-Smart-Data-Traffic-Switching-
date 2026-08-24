"""UE buffer dynamics (Guide Section 10).

Conceptual continuous-time model (the mathematical model of the system):

    dB/dt = Rp - Rd

where B is buffer level (seconds of playable content), Rp is the content
delivery/fill rate, and Rd is the content consumption/drain (playback)
rate.

The simulator is a discrete-time system, so it implements this model as
one Euler step per tick of size dt:

    B_next = max(0, B_current + (Rp - Rd) * dt)

which is the same equation as `B_next = max(0, B_current + received -
consumed)` once you name `received = Rp * dt` and `consumed = Rd * dt` —
i.e. the per-tick integrals of the fill and drain rates over the
timestep. The discrete equation below is the simulator's implementation
of the continuous model above, not a separate or replacement rule.
"""

from __future__ import annotations

from config.config import SimConfig
from traffic_sim.simulation.user import UE


def update_buffer(ue: UE, content_received_seconds: float, playback_consumed_seconds: float) -> float:
    """B_next = max(0, B_current + received - consumed)."""
    ue.buffer_seconds = max(0.0, ue.buffer_seconds + content_received_seconds - playback_consumed_seconds)
    return ue.buffer_seconds


def buffer_from_rates(ue: UE, fill_rate_seconds_per_sec: float, drain_rate_seconds_per_sec: float, dt: float) -> float:
    """Equivalent entry point when working directly with rates Rp, Rd and timestep dt.

    B_next = max(0, B_current + (Rp - Rd) * dt)
    """
    received = fill_rate_seconds_per_sec * dt
    consumed = drain_rate_seconds_per_sec * dt
    return update_buffer(ue, received, consumed)


def transition_safe(ue: UE, config: SimConfig) -> bool:
    """TransitionSafe = buffer >= transition_time + safety_margin."""
    return ue.buffer_seconds >= config.transition_time_seconds + config.safety_margin_seconds


def group_safety_buffer(ues: list[UE], percentile: float = 0.05) -> float:
    """Low-percentile (default: near-minimum) buffer across a group.

    Deliberately NOT the mean: a single dangerously-low-buffer UE can be
    hidden behind many well-buffered UEs if averaged, per the guide's
    explicit warning (Section 10.3).
    """
    if not ues:
        return 0.0
    values = sorted(ue.buffer_seconds for ue in ues)
    if len(values) == 1:
        return values[0]
    index = int(round(percentile * (len(values) - 1)))
    return values[index]
