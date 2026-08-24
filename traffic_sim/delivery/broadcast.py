"""Broadcast delivery (Guide Step 15, Section 9.5-9.6).

Resource demand is independent of group size: R_broadcast ≈ R_segment.
Per-UE SINR is still computed for every group member and the WORST-CASE
(minimum) is used for service-continuity/QoE reporting — a single
well-covered UE cannot hide a poorly-covered one. All UEs in the group
receive the same content_received_seconds this tick, sized from
resource-accounting (segment_resource_mbps vs broadcast capacity), not
from raw Shannon capacity — see traffic_sim.delivery.unicast's module
docstring for why: the guide's own interference constant makes Shannon
capacity collapse to ~0 for any non-trivial active-user count, which
would make both delivery modes indistinguishable rather than showing the
resource-driven tradeoff Section 9's KPIs are meant to demonstrate.
"""

from __future__ import annotations

import math

from config.config import SimConfig
from traffic_sim.content.segment import Segment
from traffic_sim.delivery.result import DeliveryResult
from traffic_sim.network.capacity import shannon_capacity_bps
from traffic_sim.network.path_loss import path_loss_db
from traffic_sim.network.received_power import received_power_dbm
from traffic_sim.network.resources import congestion as compute_congestion
from traffic_sim.network.resources import required_resource_broadcast
from traffic_sim.network.sinr import compute_sinr
from traffic_sim.simulation.buffer import update_buffer
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import DeliveryMode, UE

MBPS_TO_MBYTES_PER_S = 1.0 / 8.0


class BroadcastDelivery:
    def deliver(
        self,
        ues: list[UE],
        segment: Segment,
        cell: Cell,
        config: SimConfig,
        dt: float,
        active_users_in_cell: int,
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
        for ue in ues:
            distance_km = math.hypot(ue.x_km - cell.center_x_km, ue.y_km - cell.center_y_km)
            pl_db = path_loss_db(distance_km, config.path_loss_const_a, config.path_loss_const_b)
            rx_dbm = received_power_dbm(config.tx_power_dbm, pl_db)
            sinr_linear = compute_sinr(
                rx_dbm, active_users_in_cell, config.noise_floor_dbm, config.interference_coeff
            )
            sinr_by_ue[ue.ue_id] = sinr_linear
            ue.sinr_linear = sinr_linear

        # Computed for reporting/QoE/service-continuity only — see module
        # docstring for why it deliberately does not gate achievable_mbps.
        worst_case_sinr = min(sinr_by_ue.values())
        shannon_capacity_bps(config.broadcast_bandwidth_hz, worst_case_sinr)

        achievable_mbps = max(min(config.segment_resource_mbps, broadcast_capacity), 0.0)

        mbytes_per_s = achievable_mbps * MBPS_TO_MBYTES_PER_S
        seconds_of_content_per_mb = segment.duration_sec / segment.size_mb if segment.size_mb > 0 else 0.0
        received_seconds = mbytes_per_s * dt * seconds_of_content_per_mb
        consumed_seconds = config.playback_consumption_rate * dt

        for ue in ues:
            update_buffer(ue, received_seconds, consumed_seconds)
            ue.delivery_mode = DeliveryMode.BROADCAST
            result.content_received_seconds_by_ue[ue.ue_id] = received_seconds
            result.delivered_ue_ids.append(ue.ue_id)

        result.sinr_by_ue = sinr_by_ue
        return result
