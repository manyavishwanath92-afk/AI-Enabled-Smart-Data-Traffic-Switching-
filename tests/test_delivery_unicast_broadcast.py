import random

import pytest

from config.config import SimConfig
from traffic_sim.content.segment import Segment
from traffic_sim.delivery.broadcast import BroadcastDelivery
from traffic_sim.delivery.unicast import UnicastDelivery
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import DeliveryMode, UE


def make_group(n: int, cell: Cell) -> list[UE]:
    rng = random.Random(7)
    ues = []
    for i in range(n):
        ues.append(
            UE(
                ue_id=f"ue_{i}",
                x_km=cell.center_x_km + rng.uniform(-0.1, 0.1),
                y_km=cell.center_y_km + rng.uniform(-0.1, 0.1),
                cell_id=cell.cell_id,
            )
        )
    return ues


def test_unicast_required_resource_scales_with_group_size():
    cfg = SimConfig.default()
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=1000.0, broadcast_capacity_mbps=600.0)
    segment = Segment(segment_id="s0", size_mb=5.0, duration_sec=10.0)

    small_group = make_group(5, cell)
    large_group = make_group(50, cell)

    small_result = UnicastDelivery().deliver(small_group, segment, cell, cfg, dt=1.0, active_users_in_cell=5)
    large_result = UnicastDelivery().deliver(large_group, segment, cell, cfg, dt=1.0, active_users_in_cell=50)

    expected_ratio = len(large_group) / len(small_group)
    assert large_result.required_resource_mbps == pytest.approx(
        small_result.required_resource_mbps * expected_ratio
    )


def test_broadcast_required_resource_independent_of_group_size():
    cfg = SimConfig.default()
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=1000.0, broadcast_capacity_mbps=600.0)
    segment = Segment(segment_id="s0", size_mb=5.0, duration_sec=10.0)

    small_group = make_group(5, cell)
    large_group = make_group(50, cell)

    small_result = BroadcastDelivery().deliver(small_group, segment, cell, cfg, dt=1.0, active_users_in_cell=5)
    large_result = BroadcastDelivery().deliver(large_group, segment, cell, cfg, dt=1.0, active_users_in_cell=50)

    assert small_result.required_resource_mbps == large_result.required_resource_mbps == cfg.segment_resource_mbps


def test_unicast_resource_exceeds_broadcast_for_large_group():
    cfg = SimConfig.default()
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=1000.0, broadcast_capacity_mbps=600.0)
    segment = Segment(segment_id="s0", size_mb=5.0, duration_sec=10.0)
    group = make_group(50, cell)

    unicast_result = UnicastDelivery().deliver(group, segment, cell, cfg, dt=1.0, active_users_in_cell=50)
    broadcast_result = BroadcastDelivery().deliver(group, segment, cell, cfg, dt=1.0, active_users_in_cell=50)

    assert unicast_result.required_resource_mbps > broadcast_result.required_resource_mbps


def test_no_broadcast_resource_advantage_for_a_single_requester():
    """The dispersed/few-users case, made exact: with exactly one
    requesting UE, R_unicast = 1 * R_segment = R_broadcast — a pure tie,
    not a broadcast advantage. This is the deterministic counterpart to
    the guide's "few users -> unicast" guidance (Section 18.2): broadcast
    only pays off once a shared audience exists."""
    cfg = SimConfig.default()
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=1000.0, broadcast_capacity_mbps=600.0)
    segment = Segment(segment_id="s0", size_mb=5.0, duration_sec=10.0)
    single_ue = make_group(1, cell)

    unicast_result = UnicastDelivery().deliver(single_ue, segment, cell, cfg, dt=1.0, active_users_in_cell=1)
    broadcast_result = BroadcastDelivery().deliver(single_ue, segment, cell, cfg, dt=1.0, active_users_in_cell=1)

    assert unicast_result.required_resource_mbps == pytest.approx(broadcast_result.required_resource_mbps)
    assert unicast_result.required_resource_mbps == pytest.approx(cfg.segment_resource_mbps)


def test_unicast_congestion_exceeds_broadcast_when_cell_capacity_limited():
    cfg = SimConfig.default().with_overrides(segment_resource_mbps=5.0)
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=50.0, broadcast_capacity_mbps=50.0)
    segment = Segment(segment_id="s0", size_mb=5.0, duration_sec=10.0)
    group = make_group(30, cell)  # 30 * 5 = 150 Mbps demand >> 50 Mbps capacity

    unicast_result = UnicastDelivery().deliver(group, segment, cell, cfg, dt=1.0, active_users_in_cell=30)
    broadcast_result = BroadcastDelivery().deliver(group, segment, cell, cfg, dt=1.0, active_users_in_cell=30)

    assert unicast_result.congestion > broadcast_result.congestion
    assert unicast_result.congestion == 1.0  # clamped, 150 >> 50


def test_delivery_updates_ue_buffers_and_delivery_mode():
    cfg = SimConfig.default()
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=1000.0, broadcast_capacity_mbps=600.0)
    segment = Segment(segment_id="s0", size_mb=2.0, duration_sec=10.0)
    group = make_group(3, cell)
    for ue in group:
        ue.buffer_seconds = 5.0

    UnicastDelivery().deliver(group, segment, cell, cfg, dt=1.0, active_users_in_cell=3)

    for ue in group:
        assert ue.buffer_seconds >= 0.0
        assert ue.delivery_mode == DeliveryMode.UNICAST
