"""Resource demand and congestion accounting (Guide Section 9.6, 9.7).

The central principle: unicast resource demand scales with the number of
recipients; a shared broadcast transmission does not.

    R_unicast ≈ N_requesting * R_segment
    R_broadcast ≈ R_segment
    congestion = min(1, RequiredResource / AvailableResource)
"""

from __future__ import annotations


def required_resource_unicast(n_requesting: int, segment_resource_mbps: float) -> float:
    """R_unicast ≈ N_requesting * R_segment."""
    return n_requesting * segment_resource_mbps


def required_resource_broadcast(segment_resource_mbps: float) -> float:
    """R_broadcast ≈ R_segment (independent of group size)."""
    return segment_resource_mbps


def congestion(required_mbps: float, available_mbps: float) -> float:
    """congestion = min(1, required / available)."""
    if available_mbps <= 0:
        return 1.0
    return min(1.0, required_mbps / available_mbps)


def resource_savings(resource_ai: float, resource_always_unicast: float) -> float:
    """Savings = 1 - Resource_AI / Resource_always_unicast (Guide Section 9.8).

    In Phase 1 there is no AI arm; callers use this to compare a fixed
    mode_selector's resource use against a parallel always-unicast
    baseline computed for the same tick, which is enough to demonstrate
    step 20's "sensible differences" requirement without any AI existing
    yet.
    """
    if resource_always_unicast <= 0:
        return 0.0
    return 1.0 - (resource_ai / resource_always_unicast)
