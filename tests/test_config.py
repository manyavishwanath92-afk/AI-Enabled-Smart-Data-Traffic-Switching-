import pytest

from config.config import SimConfig


def test_defaults_are_valid():
    cfg = SimConfig.default()
    cfg.validate()  # should not raise


def test_interference_rise_coeff_default_and_configurable():
    cfg = SimConfig.default()
    assert cfg.interference_rise_coeff == 0.02
    cfg2 = cfg.with_overrides(interference_rise_coeff=0.1)
    assert cfg2.interference_rise_coeff == 0.1
    assert cfg.interference_rise_coeff == 0.02  # original untouched (dataclasses.replace)


def test_transition_and_buffer_fields_configurable():
    cfg = SimConfig.default()
    assert cfg.transition_time_seconds == 2.0
    assert cfg.initial_buffer_seconds == 5.0
    cfg2 = cfg.with_overrides(transition_time_seconds=5.0, initial_buffer_seconds=10.0)
    assert cfg2.transition_time_seconds == 5.0
    assert cfg2.initial_buffer_seconds == 10.0


def test_reserved_phase2_fields_present_but_inert():
    cfg = SimConfig.default()
    assert cfg.hysteresis_margin is None
    assert cfg.cooldown_period_s is None
    assert cfg.ai_model_path is None
    assert cfg.utility_weights is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_cells": 0},
        {"num_ues": -1},
        {"cellular_bandwidth_hz": 0},
        {"interference_rise_coeff": -0.1},
        {"initial_buffer_seconds": -1.0},
    ],
)
def test_invalid_configs_raise(overrides):
    cfg = SimConfig.default()
    with pytest.raises(ValueError):
        cfg.with_overrides(**overrides)
