"""Unicast delivery (Guide Step 14, Section 9.5-9.6).

Resource demand scales with the number of recipients:
R_unicast ≈ N_requesting * R_segment, and congestion is computed from
that demand against the cell's cellular capacity — unchanged from Phase 1.

What Phase 2 changes is the throughput each UE actually achieves. The
full radio chain now gates delivery instead of being computed and
discarded:

    distance -> path loss -> received power -> SINR -> Shannon capacity
             -> achievable rate -> seconds of content -> buffer

Each unicast UE shares the cell's time-frequency resources with the other
UEs the cell is serving by unicast this tick, so BOTH its radio-limited
Shannon capacity and the cell's resource capacity are divided by that
count (see network.throughput.unicast_achievable_mbps). A UE at the cell
edge, or in a heavily loaded cell, therefore receives measurably less
content and its buffer drains — which is exactly the effect the Phase 2
network model is meant to expose.

`unicast_users_in_cell` is supplied by the Simulator, which decides every
group's delivery mode before delivering any of them; UEs offloaded to
broadcast are excluded from the count, so broadcast offload genuinely
frees cellular capacity for the UEs that remain on unicast.
"""

from __future__ import annotations

from config.config import SimConfig
from traffic_sim.content.segment import Segment
from traffic_sim.delivery.result import DeliveryResult
from traffic_sim.network.radio import compute_ue_radio_state
from traffic_sim.network.resources import congestion as compute_congestion
from traffic_sim.network.resources import required_resource_unicast
from traffic_sim.network.throughput import delivered_seconds_of_content, unicast_achievable_mbps
from traffic_sim.simulation.buffer import update_buffer
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import DeliveryMode, UE


class UnicastDelivery:
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
        required = required_resource_unicast(n_requesting, config.segment_resource_mbps)
        cong = compute_congestion(required, cell.maximum_capacity_mbps)

        # Falls back to the group size when the Simulator does not supply a
        # cell-wide unicast count (e.g. direct unit-test use of this class).
        sharing_users = unicast_users_in_cell if unicast_users_in_cell is not None else n_requesting

        result = DeliveryResult(
            mode="unicast",
            segment_id=segment.segment_id,
            cell_id=cell.cell_id,
            n_requesting=n_requesting,
            required_resource_mbps=required,
            congestion=cong,
        )

        consumed_seconds = config.playback_consumption_rate * dt

        for ue in ues:
            radio = compute_ue_radio_state(
                ue, cell, config, active_users_in_cell, config.cellular_bandwidth_hz
            )
            achievable_mbps = unicast_achievable_mbps(
                shannon_mbps=radio.shannon_mbps,
                unicast_users_in_cell=sharing_users,
                cell_capacity_mbps=cell.maximum_capacity_mbps,
                segment_bitrate_mbps=segment.bitrate_mbps,
            )
            received_seconds = delivered_seconds_of_content(achievable_mbps, segment.bitrate_mbps, dt)

            update_buffer(ue, received_seconds, consumed_seconds)
            ue.sinr_linear = radio.sinr_linear
            ue.delivery_mode = DeliveryMode.UNICAST

            result.sinr_by_ue[ue.ue_id] = radio.sinr_linear
            result.achievable_mbps_by_ue[ue.ue_id] = achievable_mbps
            result.content_received_seconds_by_ue[ue.ue_id] = received_seconds
            result.delivered_ue_ids.append(ue.ue_id)

        return result
