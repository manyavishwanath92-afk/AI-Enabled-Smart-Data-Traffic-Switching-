"""Delivery latency estimation (Guide Section 11)."""

from __future__ import annotations


def estimated_full_delivery_latency_s(received_seconds_this_tick: float, segment_duration_sec: float, dt: float) -> float:
    """Estimate seconds needed to fully deliver a segment at this tick's rate.

    rate = (seconds of content received this tick) / dt, i.e. how many
    seconds of playable content arrive per second of real time at the
    currently achieved throughput. Returns inf if nothing is being
    delivered.
    """
    if received_seconds_this_tick <= 0 or dt <= 0:
        return float("inf")
    rate = received_seconds_this_tick / dt
    if rate <= 0:
        return float("inf")
    return segment_duration_sec / rate
