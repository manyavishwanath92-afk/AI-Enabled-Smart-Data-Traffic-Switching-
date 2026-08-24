"""Unicast delivery (Guide Step 14, Section 9.5-9.6).

Resource demand scales with the number of recipients:
R_unicast ≈ N_requesting * R_segment. Congestion is computed from that
demand against the cell's cellular capacity. Each UE's actual achievable
throughput this tick is capped by the segment's nominal rate
(segment_resource_mbps) and its fair share of the cell's capacity when
the cell is congested — this is exactly the guide's invitation
(Section 9.6) to "adjust the simple resource-demand expressions for
modeled capacity, coverage, reliability, overhead and transition cost."

NOTE on Shannon capacity: SINR and Shannon capacity are still computed
per UE every tick (Guide Steps 9-13) and reported on the UE / in
DeliveryResult for service-continuity and QoE purposes. They are
deliberately NOT used to further cap buffer-fill throughput here. Under
the guide's own carried-over old-PoC interference constant
(interference_linear = interference_coeff * active_users, computed
directly in linear mW — see network.sinr), interference dwarfs any
realistic received signal power converted from typical cellular dBm
values for any non-trivial active-user count, collapsing Shannon
capacity to ~0 regardless of mode. The guide itself flags this
constant as "synthetic, not a physical interference model" (Section
8.4). Gating delivery on it would make unicast and broadcast
indistinguishable (both ~0 throughput) rather than showing the
resource-driven tradeoff the guide's Section 9 KPIs are meant to
demonstrate — see Guide Section 9.4's own distinction between
theoretical channel capacity and actual delivered data rate.
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
from traffic_sim.network.resources import required_resource_unicast
from traffic_sim.network.sinr import compute_sinr
from traffic_sim.simulation.buffer import update_buffer
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import DeliveryMode, UE

MBPS_TO_MBYTES_PER_S = 1.0 / 8.0  # megabits/s -> megabytes/s


class UnicastDelivery:
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
        required = required_resource_unicast(n_requesting, config.segment_resource_mbps)
        cong = compute_congestion(required, cell.maximum_capacity_mbps)

        fair_share_mbps = (
            cell.maximum_capacity_mbps / n_requesting if n_requesting > 0 else config.segment_resource_mbps
        )

        result = DeliveryResult(
            mode="unicast",
            segment_id=segment.segment_id,
            cell_id=cell.cell_id,
            n_requesting=n_requesting,
            required_resource_mbps=required,
            congestion=cong,
        )

        for ue in ues:
            distance_km = math.hypot(ue.x_km - cell.center_x_km, ue.y_km - cell.center_y_km)
            pl_db = path_loss_db(distance_km, config.path_loss_const_a, config.path_loss_const_b)
            rx_dbm = received_power_dbm(config.tx_power_dbm, pl_db)
            sinr_linear = compute_sinr(
                rx_dbm, active_users_in_cell, config.noise_floor_dbm, config.interference_coeff
            )
            # Computed for reporting/QoE/service-continuity only — see module
            # docstring for why it deliberately does not gate achievable_mbps.
            shannon_capacity_bps(config.cellular_bandwidth_hz, sinr_linear)

            achievable_mbps = max(min(config.segment_resource_mbps, fair_share_mbps), 0.0)

            mbytes_per_s = achievable_mbps * MBPS_TO_MBYTES_PER_S
            seconds_of_content_per_mb = segment.duration_sec / segment.size_mb if segment.size_mb > 0 else 0.0
            received_seconds = mbytes_per_s * dt * seconds_of_content_per_mb
            consumed_seconds = config.playback_consumption_rate * dt

            update_buffer(ue, received_seconds, consumed_seconds)
            ue.sinr_linear = sinr_linear
            ue.delivery_mode = DeliveryMode.UNICAST

            result.sinr_by_ue[ue.ue_id] = sinr_linear
            result.content_received_seconds_by_ue[ue.ue_id] = received_seconds
            result.delivered_ue_ids.append(ue.ue_id)

        return result
