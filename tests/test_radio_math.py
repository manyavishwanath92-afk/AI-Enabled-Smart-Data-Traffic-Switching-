"""Worked-example tests taken directly from the build guide (Section 8, 9.3),
plus the noise-rise interference model that replaced the guide's original
dimensionally-unsound formula (Section 8.4)."""

import math

import pytest

from traffic_sim.network.capacity import shannon_capacity_bps
from traffic_sim.network.path_loss import path_loss_db
from traffic_sim.network.received_power import received_power_dbm
from traffic_sim.network.sinr import (
    compute_sinr,
    dbm_to_linear_mw,
    interference_noise_rise_linear,
    service_continuity_ratio,
    sinr_threshold_linear,
    sinr_to_db,
)

A, B = 128.1, 37.6  # guide's path-loss constants
TX, NOISE_FLOOR = 46.0, -104.0


def _sinr_db(distance_km: float, active_users: int, coeff: float = 0.02) -> float:
    rx = received_power_dbm(TX, path_loss_db(distance_km, A, B))
    return sinr_to_db(compute_sinr(rx, active_users, NOISE_FLOOR, coeff))


def test_path_loss_at_1km_equals_a_constant():
    # PL(1km) = a + b*log10(1) = a + 0 = a = 128.1
    assert path_loss_db(1.0, A, B) == pytest.approx(128.1)


def test_received_power_worked_example_minus_82_1_dbm():
    pl = path_loss_db(1.0, A, B)
    rx = received_power_dbm(46.0, pl)
    assert rx == pytest.approx(-82.1, abs=1e-9)


def test_path_loss_increases_with_distance():
    assert path_loss_db(2.0, A, B) > path_loss_db(1.0, A, B)


def test_dbm_to_linear_mw_conversion():
    assert dbm_to_linear_mw(0.0) == pytest.approx(1.0)
    assert dbm_to_linear_mw(10.0) == pytest.approx(10.0)
    assert dbm_to_linear_mw(-10.0) == pytest.approx(0.1)


def test_sinr_threshold_linear_worked_example():
    # 5 dB -> ~3.16 linear
    assert sinr_threshold_linear(5.0) == pytest.approx(3.1623, rel=1e-3)


def test_service_continuity_ratio():
    threshold = sinr_threshold_linear(5.0)
    observations = [threshold + 1, threshold - 1, threshold, threshold * 2]
    ratio = service_continuity_ratio(observations, threshold)
    assert ratio == pytest.approx(3 / 4)


def test_shannon_capacity_worked_example_20_5_mbps():
    # B=10MHz, SINR=3.16 -> C ~= 20.5 Mbps
    capacity_bps = shannon_capacity_bps(10_000_000, 3.16)
    assert capacity_bps / 1_000_000 == pytest.approx(20.5, rel=0.02)


def test_shannon_capacity_zero_sinr_is_zero():
    assert shannon_capacity_bps(10_000_000, 0.0) == pytest.approx(0.0)


def test_sinr_to_db_roundtrip():
    assert sinr_to_db(1.0) == pytest.approx(0.0)
    assert sinr_to_db(10.0) == pytest.approx(10.0)
    assert math.isinf(sinr_to_db(0.0))


# --- Noise-rise interference model -------------------------------------------


def test_interference_uses_configured_coefficient_not_hardcoded():
    noise = dbm_to_linear_mw(-104.0)
    # 0.02 * 30 users = 0.6 noise-floors' worth of interference
    assert interference_noise_rise_linear(30, 0.02, noise) == pytest.approx(0.6 * noise)
    # Changing the coefficient alone changes the result — nothing else needed.
    assert interference_noise_rise_linear(30, 0.04, noise) == pytest.approx(1.2 * noise)


def test_compute_sinr_uses_linear_arithmetic_only():
    sinr = compute_sinr(rx_power_dbm=-82.1, active_users=10, noise_floor_dbm=-104.0, interference_rise_coeff=0.02)
    signal_linear = dbm_to_linear_mw(-82.1)
    noise_linear = dbm_to_linear_mw(-104.0)
    interference = interference_noise_rise_linear(10, 0.02, noise_linear)
    expected = signal_linear / (interference + noise_linear)
    assert sinr == pytest.approx(expected)


def test_sinr_gives_a_realistic_value_not_a_collapsed_one():
    """SINR at a typical mid-cell distance/load must land in a usable band —
    the concrete counter-example to the removed bare-linear-mW formula,
    which collapsed SINR to ~1e-9 (about -90 dB) regardless of conditions."""
    sinr_db = _sinr_db(1.0, 30)
    assert 15 < sinr_db < 25


@pytest.mark.parametrize("active_users", [10, 30, 100])
def test_sinr_falls_monotonically_with_distance(active_users):
    distances = [0.3, 1.0, 1.5, 2.0]
    values = [_sinr_db(d, active_users) for d in distances]
    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize("distance_km", [0.3, 1.0, 2.0])
def test_sinr_falls_monotonically_with_cell_load(distance_km):
    loads = [10, 30, 60, 100]
    values = [_sinr_db(distance_km, n) for n in loads]
    assert values == sorted(values, reverse=True)


def test_sinr_stays_in_a_usable_band_across_the_operating_range():
    """Across the whole distance/load matrix SINR must stay in a band where
    the 5 dB continuity threshold is meaningful — sometimes comfortably met,
    never absurdly high or collapsed."""
    values = [_sinr_db(d, n) for d in (0.3, 1.0, 1.5, 2.0) for n in (10, 30, 60, 100)]
    assert min(values) > 5.0
    assert max(values) < 45.0
