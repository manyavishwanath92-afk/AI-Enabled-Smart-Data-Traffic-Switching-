"""Delivery reliability and service-continuity KPIs (Guide Section 11).

Service continuity is defined here, and ONLY here, as actual playback
continuity:

    SC = 1 - (total playback interruption time / total playback time)

It is not a radio measure. SINR reaches it only through the delivery
chain — SINR sets Shannon capacity, which sets the achievable delivery
rate, which is compared against the segment's bitrate to fill or drain
the UE buffer; an empty buffer is a playback stall, and stalls are what
SC counts. A separate SINR-threshold continuity metric would double-count
radio quality, so none exists. Raw radio quality is reported on its own
as the `mean_sinr_db` KPI.
"""

from __future__ import annotations

from traffic_sim.simulation.user import UE


def successful_delivery_ratio(successful_deliveries: int, required_deliveries: int) -> float:
    """successful required deliveries / required deliveries."""
    if required_deliveries <= 0:
        return 1.0
    return successful_deliveries / required_deliveries


def buffer_starvation_rate(ues: list[UE]) -> float:
    """Fraction of UEs whose buffer has reached zero (starvation) this tick."""
    if not ues:
        return 0.0
    starved = sum(1 for ue in ues if ue.buffer_seconds <= 0.0)
    return starved / len(ues)


def observe_playback(ues: list[UE]) -> None:
    """Advance each UE's playback-continuity counters for this tick.

    A UE is stalled on a tick when its buffer is empty. A rebuffering
    event is counted only on the transition from playing to stalled, so a
    long outage is one event rather than one per tick.
    """
    for ue in ues:
        ue.playback_ticks += 1
        stalled_now = ue.buffer_seconds <= 0.0
        if stalled_now:
            ue.stall_ticks += 1
            if not ue.is_stalled:
                ue.rebuffering_events += 1
        ue.is_stalled = stalled_now


def service_continuity(ues: list[UE]) -> float:
    """SC = 1 - (total playback interruption time / total playback time).

    Time is counted in UE-ticks: each UE contributes one tick of playback
    time per observed tick, and one tick of interruption time for each
    tick it spent stalled. Returns 1.0 when nothing has been observed yet
    (no playback attempted means no interruption).
    """
    total_playback_ticks = sum(ue.playback_ticks for ue in ues)
    if total_playback_ticks <= 0:
        return 1.0
    interruption_ticks = sum(ue.stall_ticks for ue in ues)
    return 1.0 - (interruption_ticks / total_playback_ticks)
