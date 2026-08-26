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


def test_cell_edge_ue_receives_less_content_than_cell_centre_ue():
    """Phase 2's core guarantee: radio conditions now gate delivery.

    Two UEs, identical in every way except distance from the cell centre,
    delivered in the same call with the same resources. The edge UE has
    worse SINR, hence lower Shannon capacity, hence a lower achievable
    rate, hence less content received. In Phase 1 these were identical
    because throughput was resource-only.
    """
    cfg = SimConfig.default()
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=1000.0, broadcast_capacity_mbps=600.0)
    segment = Segment(segment_id="s0", size_mb=5.0, duration_sec=10.0)

    centre_ue = UE(ue_id="centre", x_km=0.01, y_km=0.0, cell_id="c0")
    edge_ue = UE(ue_id="edge", x_km=3.0, y_km=0.0, cell_id="c0")

    result = UnicastDelivery().deliver(
        [centre_ue, edge_ue], segment, cell, cfg, dt=1.0, active_users_in_cell=80, unicast_users_in_cell=80
    )

    assert result.sinr_by_ue["centre"] > result.sinr_by_ue["edge"]
    assert result.achievable_mbps_by_ue["centre"] > result.achievable_mbps_by_ue["edge"]
    assert (
        result.content_received_seconds_by_ue["centre"] > result.content_received_seconds_by_ue["edge"]
    )


def test_unicast_content_received_falls_as_cell_unicast_load_rises():
    """More UEs sharing the cell -> less content each, via the radio share."""
    cfg = SimConfig.default()
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=1000.0, broadcast_capacity_mbps=600.0)
    segment = Segment(segment_id="s0", size_mb=5.0, duration_sec=10.0)

    def received_with_load(load: int) -> float:
        ue = make_group(1, cell)[0]
        result = UnicastDelivery().deliver(
            [ue], segment, cell, cfg, dt=1.0, active_users_in_cell=load, unicast_users_in_cell=load
        )
        return result.content_received_seconds_by_ue[ue.ue_id]

    # Below ~100 UEs the per-UE radio share still exceeds the segment's
    # 4 Mbps bitrate, so the cap binds and content received is flat at
    # real-time; above it, the radio share is what binds and delivery falls.
    assert received_with_load(5) == pytest.approx(received_with_load(50))
    assert received_with_load(400) < received_with_load(200) < received_with_load(100)


def test_broadcast_content_received_is_identical_for_every_group_member():
    """One shared transmission -> everyone gets the same seconds of content."""
    cfg = SimConfig.default()
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=1000.0, broadcast_capacity_mbps=600.0)
    segment = Segment(segment_id="s0", size_mb=5.0, duration_sec=10.0)
    group = make_group(20, cell)

    result = BroadcastDelivery().deliver(group, segment, cell, cfg, dt=1.0, active_users_in_cell=20)

    values = set(result.content_received_seconds_by_ue.values())
    assert len(values) == 1


def test_broadcast_beats_unicast_on_content_delivered_under_heavy_load():
    """The physical mechanism behind the whole project, at delivery level."""
    cfg = SimConfig.default()
    segment = Segment(segment_id="s0", size_mb=5.0, duration_sec=10.0)

    def fresh_cell():
        return Cell(
            cell_id="c0", center_x_km=0, center_y_km=0,
            maximum_capacity_mbps=1000.0, broadcast_capacity_mbps=600.0,
        )

    cell_u, cell_b = fresh_cell(), fresh_cell()
    group_u, group_b = make_group(80, cell_u), make_group(80, cell_b)

    u = UnicastDelivery().deliver(
        group_u, segment, cell_u, cfg, dt=1.0, active_users_in_cell=80, unicast_users_in_cell=80
    )
    b = BroadcastDelivery().deliver(group_b, segment, cell_b, cfg, dt=1.0, active_users_in_cell=80)

    mean_u = sum(u.content_received_seconds_by_ue.values()) / len(group_u)
    mean_b = sum(b.content_received_seconds_by_ue.values()) / len(group_b)
    assert mean_b > mean_u


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
