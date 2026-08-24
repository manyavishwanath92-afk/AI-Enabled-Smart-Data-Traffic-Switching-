"""Worked-example tests taken directly from the build guide (Section 8, 9.3)."""

import pytest

from traffic_sim.network.capacity import shannon_capacity_bps
from traffic_sim.network.path_loss import path_loss_db
from traffic_sim.network.received_power import received_power_dbm
from traffic_sim.network.sinr import (
    compute_sinr,
    dbm_to_linear_mw,
    interference_linear,
    service_continuity_ratio,
    sinr_threshold_linear,
)

A, B = 128.1, 37.6  # guide's path-loss constants


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


def test_interference_uses_configured_coefficient_not_hardcoded():
    # Guide's default: I_linear = 0.02 * active_users
    assert interference_linear(active_users=50, interference_coeff=0.02) == pytest.approx(1.0)
    # Changing the coefficient alone changes the result — nothing else needed.
    assert interference_linear(active_users=50, interference_coeff=0.04) == pytest.approx(2.0)


def test_sinr_threshold_linear_worked_example():
    # 5 dB -> ~3.16 linear
    assert sinr_threshold_linear(5.0) == pytest.approx(3.1623, rel=1e-3)


def test_compute_sinr_uses_linear_arithmetic_only():
    sinr = compute_sinr(
        rx_power_dbm=-82.1, active_users=10, noise_floor_dbm=-104.0, interference_coeff=0.02
    )
    signal_linear = dbm_to_linear_mw(-82.1)
    interference = interference_linear(10, 0.02)
    noise_linear = dbm_to_linear_mw(-104.0)
    expected = signal_linear / (interference + noise_linear)
    assert sinr == pytest.approx(expected)


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
