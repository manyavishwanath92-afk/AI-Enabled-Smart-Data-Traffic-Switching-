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
