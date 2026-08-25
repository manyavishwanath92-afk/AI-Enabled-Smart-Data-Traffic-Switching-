"""Phase 2: the achievable-rate layer that links radio capacity to the buffer."""

import pytest

from traffic_sim.network.throughput import (
    broadcast_achievable_mbps,
    delivered_seconds_of_content,
    unicast_achievable_mbps,
)

# Generous radio/resource headroom so the constraint under test is the one
# that actually binds in each case.
AMPLE_SHANNON = 500.0
AMPLE_CAPACITY = 500.0
BITRATE = 2.0


def _rate_at_load(n: int) -> float:
    return unicast_achievable_mbps(
        shannon_mbps=100.0,
        unicast_users_in_cell=n,
        cell_capacity_mbps=100.0,
        segment_bitrate_mbps=BITRATE,
    )


def test_unicast_rate_never_increases_with_more_ues_sharing_the_cell():
    rates = [_rate_at_load(n) for n in (1, 10, 50, 100, 200, 400)]
    assert rates == sorted(rates, reverse=True)


def test_unicast_rate_is_capped_by_bitrate_while_the_cell_is_uncontended():
    # 100 Mbps shared <=50 ways still exceeds the 2.0 Mbps bitrate cap, so
    # the cap — not the sharing — is what binds here.
    assert _rate_at_load(1) == pytest.approx(BITRATE)
    assert _rate_at_load(50) == pytest.approx(BITRATE)


def test_unicast_rate_falls_strictly_once_sharing_binds():
    # Above ~50 UEs the per-UE share drops below the bitrate cap and
    # sharing becomes the binding constraint.
    assert _rate_at_load(400) < _rate_at_load(200) < _rate_at_load(100) < BITRATE


def test_broadcast_rate_is_invariant_to_group_size():
    """The whole point of broadcast: one transmission, no per-UE division."""
    rate = broadcast_achievable_mbps(
        shannon_mbps_at_worst_sinr=AMPLE_SHANNON,
        broadcast_capacity_mbps=AMPLE_CAPACITY,
        segment_bitrate_mbps=BITRATE,
    )
    # Nothing in the signature even accepts a group size — assert the value
    # equals the bitrate cap regardless of how many UEs are listening.
    assert rate == pytest.approx(BITRATE)


def test_neither_mode_exceeds_the_segment_bitrate():
    assert unicast_achievable_mbps(AMPLE_SHANNON, 1, AMPLE_CAPACITY, BITRATE) == pytest.approx(BITRATE)
    assert broadcast_achievable_mbps(AMPLE_SHANNON, AMPLE_CAPACITY, BITRATE) == pytest.approx(BITRATE)


def test_unicast_is_radio_limited_when_shannon_is_the_binding_constraint():
    # 10 Mbps of radio shared 10 ways = 1.0 Mbps, below the 2.0 bitrate cap
    # and below the resource share, so radio is what binds.
    rate = unicast_achievable_mbps(
        shannon_mbps=10.0, unicast_users_in_cell=10, cell_capacity_mbps=AMPLE_CAPACITY, segment_bitrate_mbps=BITRATE
    )
    assert rate == pytest.approx(1.0)


def test_unicast_is_resource_limited_when_capacity_is_the_binding_constraint():
    rate = unicast_achievable_mbps(
        shannon_mbps=AMPLE_SHANNON, unicast_users_in_cell=10, cell_capacity_mbps=10.0, segment_bitrate_mbps=BITRATE
    )
    assert rate == pytest.approx(1.0)


def test_broadcast_is_limited_by_worst_case_sinr():
    rate = broadcast_achievable_mbps(
        shannon_mbps_at_worst_sinr=0.5, broadcast_capacity_mbps=AMPLE_CAPACITY, segment_bitrate_mbps=BITRATE
    )
    assert rate == pytest.approx(0.5)


def test_rates_are_never_negative():
    assert unicast_achievable_mbps(-5.0, 10, 100.0, BITRATE) >= 0.0
    assert broadcast_achievable_mbps(-5.0, 100.0, BITRATE) >= 0.0


def test_unicast_zero_users_does_not_divide_by_zero():
    assert unicast_achievable_mbps(AMPLE_SHANNON, 0, AMPLE_CAPACITY, BITRATE) == pytest.approx(BITRATE)


def test_receiving_exactly_the_bitrate_yields_realtime_content():
    """Full bitrate -> one second of content per second of real time."""
    assert delivered_seconds_of_content(2.0, 2.0, dt=1.0) == pytest.approx(1.0)
    assert delivered_seconds_of_content(2.0, 2.0, dt=3.0) == pytest.approx(3.0)


def test_receiving_half_the_bitrate_yields_half_realtime_content():
    """This is what makes a starved UE's buffer drain against playback."""
    assert delivered_seconds_of_content(1.0, 2.0, dt=1.0) == pytest.approx(0.5)


def test_delivered_seconds_degenerate_inputs():
    assert delivered_seconds_of_content(2.0, 0.0, dt=1.0) == 0.0
    assert delivered_seconds_of_content(2.0, 2.0, dt=0.0) == 0.0
