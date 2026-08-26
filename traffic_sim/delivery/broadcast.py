"""Broadcast delivery (Guide Step 15, Section 9.5-9.6).

Resource demand is independent of group size (R_broadcast ≈ R_segment),
and congestion is computed against the cell's broadcast capacity —
unchanged from Phase 1.

Phase 2 makes the radio chain gate delivery here too. Per-UE SINR is
computed for every group member and the transmission is sized at the
group's WORST-CASE (minimum) SINR, so every member can decode it — a
well-covered UE cannot hide a poorly-covered one. Crucially, that shared
rate is NOT divided by group size (see
network.throughput.broadcast_achievable_mbps): one transmission serves
everyone, which is the physical reason broadcast outperforms unicast
under a flash crowd, and why it degrades gracefully as the group grows
while unicast collapses.

All UEs in the group receive the same seconds of content this tick.
"""

from __future__ import annotations

from config.config import SimConfig
from traffic_sim.content.segment import Segment
from traffic_sim.delivery.result import DeliveryResult
from traffic_sim.network.radio import compute_ue_radio_state
from traffic_sim.network.resources import congestion as compute_congestion
from traffic_sim.network.resources import required_resource_broadcast
from traffic_sim.network.throughput import broadcast_achievable_mbps, delivered_seconds_of_content
from traffic_sim.simulation.buffer import update_buffer
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import DeliveryMode, UE


class BroadcastDelivery:
    def deliver(
        self,
        ues: list[UE],
        segment: Segment,
        cell: Cell,
        config: SimConfig,
        dt: float,
        active_users_in_cell: int,
        unicast_users_in_cell: int | None = None,
        broadcast_groups_in_cell: int = 1,
    ) -> DeliveryResult:
        n_requesting = len(ues)
        required = required_resource_broadcast(config.segment_resource_mbps)
        broadcast_capacity = cell.broadcast_capacity_mbps or cell.maximum_capacity_mbps
        cong = compute_congestion(required, broadcast_capacity)

        result = DeliveryResult(
            mode="broadcast",
            segment_id=segment.segment_id,
            cell_id=cell.cell_id,
            n_requesting=n_requesting,
            required_resource_mbps=required,
            congestion=cong,
        )

        if not ues:
            return result

        sinr_by_ue: dict[str, float] = {}
        worst_shannon_mbps = float("inf")
        for ue in ues:
            radio = compute_ue_radio_state(
                ue, cell, config, active_users_in_cell, config.broadcast_bandwidth_hz
            )
            sinr_by_ue[ue.ue_id] = radio.sinr_linear
            ue.sinr_linear = radio.sinr_linear
            worst_shannon_mbps = min(worst_shannon_mbps, radio.shannon_mbps)

        achievable_mbps = broadcast_achievable_mbps(
            shannon_mbps_at_worst_sinr=worst_shannon_mbps,
            broadcast_capacity_mbps=broadcast_capacity,
            segment_bitrate_mbps=segment.bitrate_mbps,
            broadcast_groups_in_cell=broadcast_groups_in_cell,
        )
        received_seconds = delivered_seconds_of_content(achievable_mbps, segment.bitrate_mbps, dt)
        consumed_seconds = config.playback_consumption_rate * dt

        for ue in ues:
            update_buffer(ue, received_seconds, consumed_seconds)
            ue.delivery_mode = DeliveryMode.BROADCAST
            result.achievable_mbps_by_ue[ue.ue_id] = achievable_mbps
            result.content_received_seconds_by_ue[ue.ue_id] = received_seconds
            result.delivered_ue_ids.append(ue.ue_id)

        result.sinr_by_ue = sinr_by_ue
        return result
