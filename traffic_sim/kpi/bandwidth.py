"""Bandwidth / spectrum-efficiency KPIs (Guide Section 11, 9.8)."""

from __future__ import annotations

from traffic_sim.network.resources import resource_savings as _resource_savings

# Re-exported so callers can import all "bandwidth" KPIs from one place;
# the formula itself lives in network.resources since it is also used
# directly by the resource-accounting layer.
resource_savings = _resource_savings


def spectrum_efficiency(delivered_bits: float, bandwidth_hz: float, observation_time_s: float) -> float:
    """Useful bits per Hz per second: delivered_bits / (bandwidth * observation_time)."""
    denom = bandwidth_hz * observation_time_s
    if denom <= 0:
        return 0.0
    return delivered_bits / denom


def redundant_payload_bytes(unicast_payload_mb: float, broadcast_payload_mb: float) -> float:
    """How much repeated payload unicast sent beyond what broadcast would have sent."""
    return max(unicast_payload_mb - broadcast_payload_mb, 0.0)
