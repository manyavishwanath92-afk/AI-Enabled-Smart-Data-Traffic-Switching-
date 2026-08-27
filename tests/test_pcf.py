"""PCF: independent policy enforcement (priority buffer margin, confidence floor)."""

import pytest

from config.config import SimConfig
from traffic_sim.policy.pcf import CONFIDENCE_FLOOR, PRIORITY_BUFFER_MARGIN, PCFPolicy
from traffic_sim.simulation.user import DeliveryMode


def make_pcf(**overrides) -> PCFPolicy:
    cfg = SimConfig.default().with_overrides(**overrides) if overrides else SimConfig.default()
    return PCFPolicy(cfg)


def test_compliant_switch_passes_through_unchanged():
    pcf = make_pcf(transition_time_seconds=2.0, safety_margin_seconds=1.0, min_decision_confidence=0.15)
    result = pcf.enforce(
        mode=DeliveryMode.BROADCAST,
        current_mode=DeliveryMode.UNICAST,
        service_priority=0,
        buffer_min_s=10.0,  # comfortably above any margin
        confidence=0.9,
    )
    assert result.final_mode == DeliveryMode.BROADCAST
    assert result.compliant is True
    assert result.violated_rules == []


def test_low_priority_content_uses_the_base_margin_only():
    pcf = make_pcf(
        transition_time_seconds=2.0, safety_margin_seconds=1.0,
        pcf_priority_threshold=1, pcf_priority_margin_multiplier=2.0,
        min_decision_confidence=0.0,
    )
    # base margin = 3.0s; buffer of 3.5s clears it but would NOT clear the
    # priority-scaled 6.0s margin, proving priority=0 doesn't get scaled.
    result = pcf.enforce(
        mode=DeliveryMode.BROADCAST, current_mode=DeliveryMode.UNICAST,
        service_priority=0, buffer_min_s=3.5, confidence=0.9,
    )
    assert result.compliant is True
    assert result.final_mode == DeliveryMode.BROADCAST


def test_high_priority_content_requires_a_larger_margin():
    pcf = make_pcf(
        transition_time_seconds=2.0, safety_margin_seconds=1.0,
        pcf_priority_threshold=1, pcf_priority_margin_multiplier=2.0,
        min_decision_confidence=0.0,
    )
    # Same 3.5s buffer, but service_priority=1 now needs 3.0*2.0=6.0s.
    result = pcf.enforce(
        mode=DeliveryMode.BROADCAST, current_mode=DeliveryMode.UNICAST,
        service_priority=1, buffer_min_s=3.5, confidence=0.9,
    )
    assert result.compliant is False
    assert PRIORITY_BUFFER_MARGIN in result.violated_rules
    assert result.final_mode == DeliveryMode.UNICAST  # reverted to incumbent


def test_priority_margin_rule_does_not_apply_when_holding_the_current_mode():
    """No transition is happening, so there is nothing to protect against."""
    pcf = make_pcf(transition_time_seconds=2.0, safety_margin_seconds=1.0, min_decision_confidence=0.0)
    result = pcf.enforce(
        mode=DeliveryMode.UNICAST, current_mode=DeliveryMode.UNICAST,
        service_priority=5, buffer_min_s=0.1, confidence=0.9,
    )
    assert result.compliant is True
    assert result.final_mode == DeliveryMode.UNICAST


def test_zero_buffer_is_exempt_so_the_stall_escape_can_still_work():
    """A buffer of exactly zero means the stream has already failed -
    blocking the switch here would only prevent recovery, recreating the
    deadlock the agent's own switch_escape_stall gate exists to avoid."""
    pcf = make_pcf(
        transition_time_seconds=2.0, safety_margin_seconds=1.0,
        pcf_priority_threshold=0, pcf_priority_margin_multiplier=5.0,
        min_decision_confidence=0.0,
    )
    result = pcf.enforce(
        mode=DeliveryMode.UNICAST, current_mode=DeliveryMode.BROADCAST,
        service_priority=3, buffer_min_s=0.0, confidence=0.9,
    )
    assert result.compliant is True
    assert result.final_mode == DeliveryMode.UNICAST


def test_low_confidence_reverts_to_incumbent_even_if_buffer_is_fine():
    pcf = make_pcf(min_decision_confidence=0.5)
    result = pcf.enforce(
        mode=DeliveryMode.BROADCAST, current_mode=DeliveryMode.UNICAST,
        service_priority=0, buffer_min_s=100.0, confidence=0.1,
    )
    assert result.compliant is False
    assert CONFIDENCE_FLOOR in result.violated_rules
    assert result.final_mode == DeliveryMode.UNICAST


def test_low_confidence_with_no_incumbent_keeps_the_proposed_mode():
    """First-ever decision for a group: there is no incumbent to revert
    to, so PCF flags the violation but cannot fall back to nothing."""
    pcf = make_pcf(min_decision_confidence=0.5)
    result = pcf.enforce(
        mode=DeliveryMode.BROADCAST, current_mode=None,
        service_priority=0, buffer_min_s=100.0, confidence=0.1,
    )
    assert result.compliant is False
    assert result.final_mode == DeliveryMode.BROADCAST


def test_confidence_floor_is_independent_of_the_agents_own_check():
    """PCF re-validates confidence itself rather than trusting the caller
    - this is the defense-in-depth the module exists to provide."""
    pcf = make_pcf(min_decision_confidence=0.2)
    # A confidence value PCF alone would reject, regardless of what any
    # upstream gate decided.
    result = pcf.enforce(
        mode=DeliveryMode.BROADCAST, current_mode=DeliveryMode.UNICAST,
        service_priority=0, buffer_min_s=100.0, confidence=0.05,
    )
    assert result.compliant is False


def test_reporting_interval_is_read_from_config_not_reimplemented():
    cfg = SimConfig.default().with_overrides(reporting_interval_s=2.5)
    pcf = PCFPolicy(cfg)
    assert pcf.reporting_interval_s == 2.5


def test_explain_returns_a_plain_serialisable_dict():
    pcf = make_pcf(min_decision_confidence=0.9)
    result = pcf.enforce(
        mode=DeliveryMode.BROADCAST, current_mode=DeliveryMode.UNICAST,
        service_priority=0, buffer_min_s=100.0, confidence=0.1,
    )
    explanation = result.explain()
    assert explanation["compliant"] is False
    assert CONFIDENCE_FLOOR in explanation["violated_rules"]
    assert set(explanation) == {"compliant", "violated_rules"}


@pytest.mark.parametrize("bad_multiplier", [0.0, 0.99])
def test_config_rejects_a_margin_multiplier_below_one(bad_multiplier):
    with pytest.raises(ValueError):
        SimConfig.default().with_overrides(pcf_priority_margin_multiplier=bad_multiplier)


def test_config_rejects_a_negative_priority_threshold():
    with pytest.raises(ValueError):
        SimConfig.default().with_overrides(pcf_priority_threshold=-1)
