"""Per-UE radio state: the front half of the Phase 2 delivery chain.

    distance -> path loss -> received power -> SINR -> Shannon capacity

Both unicast and broadcast delivery need exactly this chain per UE, and
previously duplicated it line-for-line. It is consolidated here, reusing
the existing Phase 1 primitives unchanged (network.path_loss,
network.received_power, network.sinr, network.capacity).

The back half of the chain — turning Shannon capacity into an actual
achievable delivery rate and then into seconds of content — lives in
network.throughput.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config.config import SimConfig
from traffic_sim.network.capacity import shannon_capacity_bps
from traffic_sim.network.path_loss import path_loss_db
from traffic_sim.network.received_power import received_power_dbm
from traffic_sim.network.sinr import compute_sinr
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import UE


@dataclass
class RadioState:
    distance_km: float
    path_loss_db: float
    rx_power_dbm: float
    sinr_linear: float
    shannon_mbps: float


def compute_ue_radio_state(
    ue: UE,
    cell: Cell,
    config: SimConfig,
    active_users_in_cell: int,
    bandwidth_hz: float,
) -> RadioState:
    """Run the full distance -> SINR -> Shannon chain for one UE.

    `bandwidth_hz` is the carrier the UE is being served on
    (config.cellular_bandwidth_hz for unicast,
    config.broadcast_bandwidth_hz for broadcast), so the returned
    Shannon capacity is the capacity of that carrier at this UE's SINR.
    """
    distance_km = math.hypot(ue.x_km - cell.center_x_km, ue.y_km - cell.center_y_km)
    pl_db = path_loss_db(distance_km, config.path_loss_const_a, config.path_loss_const_b)
    rx_dbm = received_power_dbm(config.tx_power_dbm, pl_db)

    sinr_linear = compute_sinr(
        rx_dbm,
        active_users_in_cell,
        config.noise_floor_dbm,
        config.interference_rise_coeff,
    )

    shannon_mbps = shannon_capacity_bps(bandwidth_hz, sinr_linear) / 1_000_000.0

    return RadioState(
        distance_km=distance_km,
        path_loss_db=pl_db,
        rx_power_dbm=rx_dbm,
        sinr_linear=sinr_linear,
        shannon_mbps=shannon_mbps,
    )
