import pytest

from config.config import SimConfig
from traffic_sim.simulation.buffer import buffer_from_rates, group_safety_buffer, transition_safe, update_buffer
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.mobility import compute_mobility_score
from traffic_sim.simulation.user import MobilityState, UE


def test_update_buffer_never_negative():
    ue = UE(ue_id="ue_1", x_km=0, y_km=0, buffer_seconds=1.0)
    update_buffer(ue, content_received_seconds=0.0, playback_consumed_seconds=5.0)
    assert ue.buffer_seconds == 0.0


def test_update_buffer_grows_when_delivery_faster_than_playback():
    ue = UE(ue_id="ue_1", x_km=0, y_km=0, buffer_seconds=5.0)
    update_buffer(ue, content_received_seconds=3.0, playback_consumed_seconds=1.0)
    assert ue.buffer_seconds == 7.0


def test_discrete_buffer_matches_continuous_rate_formulation():
    """B_next = max(0, B + received - consumed) must equal
    max(0, B + (Rp - Rd) * dt) for the same received/consumed = Rp*dt, Rd*dt."""
    ue_a = UE(ue_id="a", x_km=0, y_km=0, buffer_seconds=5.0)
    ue_b = UE(ue_id="b", x_km=0, y_km=0, buffer_seconds=5.0)

    dt = 2.0
    rp, rd = 3.0, 1.5

    update_buffer(ue_a, content_received_seconds=rp * dt, playback_consumed_seconds=rd * dt)
    buffer_from_rates(ue_b, fill_rate_seconds_per_sec=rp, drain_rate_seconds_per_sec=rd, dt=dt)

    assert ue_a.buffer_seconds == pytest.approx(ue_b.buffer_seconds)


def test_transition_safe_boundary():
    cfg = SimConfig.default().with_overrides(transition_time_seconds=2.0, safety_margin_seconds=1.0)
    ue = UE(ue_id="ue_1", x_km=0, y_km=0, buffer_seconds=3.0)
    assert transition_safe(ue, cfg) is True  # exactly at boundary
    ue.buffer_seconds = 2.999
    assert transition_safe(ue, cfg) is False


def test_group_safety_buffer_uses_low_value_not_mean():
    ues = [
        UE(ue_id="a", x_km=0, y_km=0, buffer_seconds=1.0),
        UE(ue_id="b", x_km=0, y_km=0, buffer_seconds=20.0),
        UE(ue_id="c", x_km=0, y_km=0, buffer_seconds=25.0),
    ]
    safety = group_safety_buffer(ues, percentile=0.0)
    mean = sum(u.buffer_seconds for u in ues) / len(ues)
    assert safety == 1.0
    assert safety < mean


def test_stationary_ue_has_zero_mobility_risk():
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=100.0)
    ue = UE(ue_id="ue_1", x_km=0, y_km=0, velocity_mps=0.0, mobility_state=MobilityState.STATIONARY)
    risk = compute_mobility_score(ue, cell, cell_area_radius_km=2.0)
    assert risk == 0.0


def test_fast_ue_near_edge_has_higher_risk_than_slow_ue_near_center():
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=100.0)
    fast_edge_ue = UE(ue_id="a", x_km=1.9, y_km=0, velocity_mps=5.0, mobility_state=MobilityState.VEHICULAR)
    slow_center_ue = UE(ue_id="b", x_km=0.1, y_km=0, velocity_mps=0.5, mobility_state=MobilityState.WALKING)

    fast_risk = compute_mobility_score(fast_edge_ue, cell, cell_area_radius_km=2.0)
    slow_risk = compute_mobility_score(slow_center_ue, cell, cell_area_radius_km=2.0)

    assert fast_risk > slow_risk
