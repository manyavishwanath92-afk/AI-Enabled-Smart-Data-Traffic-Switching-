"""Delivery reliability and service-continuity KPIs (Guide Section 11, 8.6)."""

from __future__ import annotations

from traffic_sim.network.sinr import service_continuity_ratio
from traffic_sim.simulation.user import UE

# Re-exported: service continuity is defined in network.sinr since it is
# fundamentally a radio-quality measure, but it is also a top-level KPI.
service_continuity = service_continuity_ratio


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


def playback_continuity(ues: list[UE]) -> float:
    """Fraction of observed UE-ticks spent actually playing rather than stalled.

    This is the "actual playback/delivery continuity" half of the guide's
    service-continuity KPI (Section 11); `service_continuity` above is the
    SINR-threshold half. Both are reported.
    """
    total_ticks = sum(ue.playback_ticks for ue in ues)
    if total_ticks <= 0:
        return 1.0
    stalled = sum(ue.stall_ticks for ue in ues)
    return 1.0 - (stalled / total_ticks)
