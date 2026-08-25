"""Shared result type returned by both delivery modes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeliveryResult:
    mode: str  # "unicast" or "broadcast"
    segment_id: str
    cell_id: str
    n_requesting: int
    required_resource_mbps: float
    congestion: float
    sinr_by_ue: dict[str, float] = field(default_factory=dict)
    # Actual achievable delivery rate per UE after the radio and resource
    # constraints in network.throughput — the Phase 2 link between the
    # radio chain and what the buffer actually receives.
    achievable_mbps_by_ue: dict[str, float] = field(default_factory=dict)
    content_received_seconds_by_ue: dict[str, float] = field(default_factory=dict)
    delivered_ue_ids: list[str] = field(default_factory=list)
