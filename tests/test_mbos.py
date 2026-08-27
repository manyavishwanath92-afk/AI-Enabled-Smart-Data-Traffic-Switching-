"""MBOS: session/bearer coordination and handover detection."""

from traffic_sim.mbos.controller import BROADCAST_BEARER, CELLULAR_BEARER, MBOSController, bearer_for
from traffic_sim.simulation.user import DeliveryMode


def test_bearer_for_maps_modes_to_physical_bearers():
    assert bearer_for(DeliveryMode.UNICAST) == CELLULAR_BEARER
    assert bearer_for(DeliveryMode.BROADCAST) == BROADCAST_BEARER


def test_coordinate_assigns_a_stable_session_id_per_group():
    mbos = MBOSController()
    a = mbos.coordinate("cell_0", "seg_1", DeliveryMode.UNICAST)
    b = mbos.coordinate("cell_0", "seg_1", DeliveryMode.BROADCAST)
    assert a.session_id == b.session_id == "cell_0:seg_1"


def test_different_groups_get_different_sessions():
    mbos = MBOSController()
    a = mbos.coordinate("cell_0", "seg_1", DeliveryMode.UNICAST)
    b = mbos.coordinate("cell_0", "seg_2", DeliveryMode.UNICAST)
    assert a.session_id != b.session_id


def test_no_handover_on_first_ever_decision():
    mbos = MBOSController()
    result = mbos.coordinate("cell_0", "seg_1", DeliveryMode.BROADCAST)
    assert result.handover is False


def test_confirm_then_coordinate_same_mode_reports_no_handover():
    mbos = MBOSController()
    mbos.coordinate("cell_0", "seg_1", DeliveryMode.UNICAST)
    mbos.confirm("cell_0", "seg_1", DeliveryMode.UNICAST)

    result = mbos.coordinate("cell_0", "seg_1", DeliveryMode.UNICAST)
    assert result.handover is False
    assert result.target_bearer == CELLULAR_BEARER


def test_confirm_then_coordinate_different_mode_reports_handover():
    mbos = MBOSController()
    mbos.coordinate("cell_0", "seg_1", DeliveryMode.UNICAST)
    mbos.confirm("cell_0", "seg_1", DeliveryMode.UNICAST)

    result = mbos.coordinate("cell_0", "seg_1", DeliveryMode.BROADCAST)
    assert result.handover is True
    assert result.target_bearer == BROADCAST_BEARER


def test_coordinate_alone_never_commits_state():
    """coordinate() only proposes; without confirm(), repeated calls must
    not accumulate handover state — this is what lets PCF veto a proposal
    without corrupting next tick's handover detection."""
    mbos = MBOSController()
    mbos.coordinate("cell_0", "seg_1", DeliveryMode.UNICAST)
    mbos.confirm("cell_0", "seg_1", DeliveryMode.UNICAST)

    # Propose broadcast but never confirm it (simulating a PCF veto).
    mbos.coordinate("cell_0", "seg_1", DeliveryMode.BROADCAST)

    # Next tick's proposal should still see UNICAST as the last confirmed
    # bearer, not the vetoed BROADCAST proposal.
    result = mbos.coordinate("cell_0", "seg_1", DeliveryMode.UNICAST)
    assert result.handover is False


def test_invalid_mode_falls_back_to_unicast_and_is_flagged_corrected():
    mbos = MBOSController()
    result = mbos.coordinate("cell_0", "seg_1", DeliveryMode.NONE)
    assert result.mode == DeliveryMode.UNICAST
    assert result.target_bearer == CELLULAR_BEARER
    assert result.corrected is True


def test_valid_mode_is_never_flagged_corrected():
    mbos = MBOSController()
    result = mbos.coordinate("cell_0", "seg_1", DeliveryMode.BROADCAST)
    assert result.corrected is False


def test_explain_returns_a_plain_serialisable_dict():
    mbos = MBOSController()
    result = mbos.coordinate("cell_0", "seg_1", DeliveryMode.BROADCAST)
    explanation = result.explain()
    assert explanation["session_id"] == "cell_0:seg_1"
    assert explanation["target_bearer"] == BROADCAST_BEARER
    assert set(explanation) == {"session_id", "target_bearer", "handover", "corrected"}
