"""Achievable delivery rate and content received: the back half of the chain.

    Shannon capacity -> achievable delivery rate -> seconds of content

This is the layer Phase 1 was missing. Previously the achievable rate was
decided purely by resource accounting and the computed Shannon capacity
was discarded, so radio conditions had no effect on what a UE actually
received. Here both constraints apply, and whichever binds, binds:

  UNICAST   rate = min(segment_bitrate,
                       shannon_mbps / unicast_users_in_cell,
                       cell_capacity_mbps / unicast_users_in_cell)

      Unicast UEs share the cell's time-frequency resources, so both the
      radio-limited capacity AND the resource-limited capacity are divided
      by the number of UEs the cell is serving by unicast this tick. Rate
      therefore degrades as the cell fills up.

  BROADCAST rate = min(segment_bitrate,
                       shannon_mbps_at_worst_sinr,
                       broadcast_capacity_mbps)

      One shared transmission serves the whole group, so there is NO
      division by group size — this is the physical reason broadcast wins
      under a flash crowd. It is sized at the group's WORST-CASE SINR so
      that every member can decode it; a well-covered UE cannot hide a
      poorly-covered one.

Neither rate exceeds the segment's own bitrate: you cannot usefully
receive a stream faster than it was encoded.
"""

from __future__ import annotations


def unicast_achievable_mbps(
    shannon_mbps: float,
    unicast_users_in_cell: int,
    cell_capacity_mbps: float,
    segment_bitrate_mbps: float,
) -> float:
    """Per-UE unicast rate: radio share and resource share, whichever binds."""
    share = max(unicast_users_in_cell, 1)
    radio_limited = shannon_mbps / share
    resource_limited = cell_capacity_mbps / share
    return max(min(segment_bitrate_mbps, radio_limited, resource_limited), 0.0)


def broadcast_achievable_mbps(
    shannon_mbps_at_worst_sinr: float,
    broadcast_capacity_mbps: float,
    segment_bitrate_mbps: float,
) -> float:
    """Shared broadcast rate — independent of how many UEs are listening."""
    return max(min(segment_bitrate_mbps, shannon_mbps_at_worst_sinr, broadcast_capacity_mbps), 0.0)


def delivered_seconds_of_content(achievable_mbps: float, segment_bitrate_mbps: float, dt: float) -> float:
    """Seconds of playable content received during a tick of length dt.

    received = (achievable_rate / segment_bitrate) * dt

    Receiving exactly the segment's bitrate yields one second of content
    per second of real time; receiving half of it yields half a second,
    and the buffer drains against real-time playback.
    """
    if segment_bitrate_mbps <= 0 or dt <= 0:
        return 0.0
    return (achievable_mbps / segment_bitrate_mbps) * dt
