"""Features, counterfactual outcomes, utility and agent behaviour."""

import pytest

from config.config import SimConfig
from traffic_sim.ai.agent import AIAgent
from traffic_sim.ai.feature_engineering import FEATURE_NAMES, extract_features
from traffic_sim.ai.outcomes import evaluate_modes
from traffic_sim.ai.utility import DEFAULT_WEIGHTS, compute_utility, resolve_weights
from traffic_sim.content.segment import Segment
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import DeliveryMode, UE


def make_cell(broadcast_capacity: float = 60.0) -> Cell:
    cell = Cell(
        cell_id="c0",
        center_x_km=0.0,
        center_y_km=0.0,
        maximum_capacity_mbps=100.0,
        broadcast_capacity_mbps=broadcast_capacity,
    )
    return cell


def make_group(n: int, cell: Cell, buffer_s: float = 5.0) -> list[UE]:
    ues = []
    for i in range(n):
        ue = UE(
            ue_id=f"ue_{i}",
            x_km=cell.center_x_km + 0.05 * (i % 5),
            y_km=cell.center_y_km,
            cell_id=cell.cell_id,
            buffer_seconds=buffer_s,
            mobility_risk=0.0,
        )
        cell.connect_ue(ue.ue_id)
        ues.append(ue)
    return ues


SEGMENT = Segment(segment_id="s0", size_mb=2.5, duration_sec=10.0, bitrate_mbps=2.0, popularity=0.5)


# --- features ---------------------------------------------------------------


def test_feature_vector_matches_declared_order_and_length():
    cfg = SimConfig.default()
    cell = make_cell()
    features = extract_features(SEGMENT, make_group(5, cell), cell, cfg)

    assert list(features.values.keys()) == FEATURE_NAMES
    assert len(features.vector()) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in features.vector())


def test_feature_extraction_does_not_mutate_state():
    """Measuring the network must not change it."""
    cfg = SimConfig.default()
    cell = make_cell()
    ues = make_group(4, cell)
    before_buffers = [ue.buffer_seconds for ue in ues]
    before_load = cell.current_load_mbps

    extract_features(SEGMENT, ues, cell, cfg)

    assert [ue.buffer_seconds for ue in ues] == before_buffers
    assert cell.current_load_mbps == before_load


def test_features_reflect_group_size_and_buffers():
    cfg = SimConfig.default()
    cell = make_cell()
    ues = make_group(6, cell, buffer_s=4.0)
    ues[0].buffer_seconds = 0.5  # one UE in trouble

    features = extract_features(SEGMENT, ues, cell, cfg)

    assert features.values["n_requesting"] == 6
    assert features.values["buffer_min_s"] == 0.5
    assert features.values["buffer_mean_s"] > features.values["buffer_min_s"]


# --- counterfactual outcomes ------------------------------------------------


def test_broadcast_beats_unicast_once_the_cell_is_too_crowded_for_unicast():
    """The group must be big enough that each unicast share falls below
    the segment bitrate; below that both modes simply hit the bitrate cap
    and tie, which is correct but uninteresting."""
    cfg = SimConfig.default()
    cell = make_cell()
    features = extract_features(SEGMENT, make_group(80, cell), cell, cfg)
    outcomes = evaluate_modes(features, cfg)

    assert outcomes.unicast.achievable_mbps < SEGMENT.bitrate_mbps
    assert outcomes.broadcast.achievable_mbps > outcomes.unicast.achievable_mbps


def test_modes_tie_when_both_can_deliver_the_full_bitrate():
    """Neither mode can beat the stream's own bitrate, so a lightly
    loaded cell is a genuine tie rather than a broadcast win."""
    cfg = SimConfig.default()
    cell = make_cell()
    outcomes = evaluate_modes(extract_features(SEGMENT, make_group(5, cell), cell, cfg), cfg)

    assert outcomes.unicast.achievable_mbps == pytest.approx(SEGMENT.bitrate_mbps)
    assert outcomes.broadcast.achievable_mbps == pytest.approx(SEGMENT.bitrate_mbps)


def test_broadcast_degrades_when_the_carrier_is_contended():
    """More concurrent broadcast streams -> a thinner slice each."""
    cfg = SimConfig.default()
    cell = make_cell(broadcast_capacity=4.0)
    ues = make_group(20, cell)

    free = evaluate_modes(extract_features(SEGMENT, ues, cell, cfg, prev_broadcast_groups=0), cfg)
    busy = evaluate_modes(extract_features(SEGMENT, ues, cell, cfg, prev_broadcast_groups=8), cfg)

    assert busy.broadcast.achievable_mbps < free.broadcast.achievable_mbps


# --- utility ----------------------------------------------------------------


def test_utility_terms_stay_in_unit_range():
    cfg = SimConfig.default()
    cell = make_cell()
    features = extract_features(SEGMENT, make_group(10, cell), cell, cfg)
    outcomes = evaluate_modes(features, cfg)

    for mode, outcome in ((DeliveryMode.UNICAST, outcomes.unicast), (DeliveryMode.BROADCAST, outcomes.broadcast)):
        breakdown = compute_utility(mode, outcome, features, cfg, None)
        for name, value in breakdown.terms.items():
            assert 0.0 <= value <= 1.0, f"{mode} term {name} out of range: {value}"


def test_efficiency_benefits_are_worthless_without_delivery():
    """A mode that cannot sustain the stream must not bank its savings.

    Two identical situations, differing only in achievable rate. The
    starved one must score lower despite identical resource savings.
    """
    cfg = SimConfig.default()
    cell = make_cell()
    features = extract_features(SEGMENT, make_group(30, cell), cell, cfg)
    outcomes = evaluate_modes(features, cfg)

    healthy = outcomes.broadcast
    starved = type(healthy)(
        achievable_mbps=0.1,
        delivered_seconds=0.05,
        projected_min_buffer_s=0.0,
        would_stall=True,
    )

    u_healthy = compute_utility(DeliveryMode.BROADCAST, healthy, features, cfg, None)
    u_starved = compute_utility(DeliveryMode.BROADCAST, starved, features, cfg, None)

    assert u_starved.total < u_healthy.total
    assert u_starved.terms["resource"] < u_healthy.terms["resource"]


def test_switching_away_from_the_current_mode_costs_something():
    cfg = SimConfig.default()
    cell = make_cell()
    features = extract_features(SEGMENT, make_group(10, cell), cell, cfg)
    outcomes = evaluate_modes(features, cfg)

    staying = compute_utility(DeliveryMode.UNICAST, outcomes.unicast, features, cfg, DeliveryMode.UNICAST)
    switching = compute_utility(DeliveryMode.UNICAST, outcomes.unicast, features, cfg, DeliveryMode.BROADCAST)

    assert switching.total < staying.total
    assert switching.terms["switch"] == 1.0


def test_weights_adapt_to_low_buffers():
    cfg = SimConfig.default()
    cell = make_cell()
    comfortable = extract_features(SEGMENT, make_group(5, cell, buffer_s=20.0), cell, cfg)
    starving = extract_features(SEGMENT, make_group(5, cell, buffer_s=0.2), cell, cfg)

    assert resolve_weights(starving, cfg)["buffer"] > resolve_weights(comfortable, cfg)["buffer"]


def test_weights_are_normalised():
    cfg = SimConfig.default()
    cell = make_cell()
    features = extract_features(SEGMENT, make_group(5, cell), cell, cfg)
    assert sum(resolve_weights(features, cfg).values()) == pytest.approx(1.0)


def test_default_weights_are_used_when_config_leaves_them_unset():
    cfg = SimConfig.default()
    assert cfg.utility_weights is None
    cell = make_cell()
    features = extract_features(SEGMENT, make_group(5, cell, buffer_s=20.0), cell, cfg)
    weights = resolve_weights(features, cfg)
    assert set(weights) == set(DEFAULT_WEIGHTS)


# --- agent ------------------------------------------------------------------


def test_agent_matches_the_mode_selector_signature():
    cfg = SimConfig.default()
    agent = AIAgent(cfg, model_dir="does-not-exist")
    cell = make_cell()
    mode = agent("s0", make_group(5, cell), cell, cfg)
    assert isinstance(mode, DeliveryMode)


def test_agent_falls_back_gracefully_without_a_model():
    cfg = SimConfig.default()
    agent = AIAgent(cfg, model_dir="does-not-exist")
    assert agent.uses_model is False

    cell = make_cell()
    decision = agent.decide("s0", make_group(5, cell), cell, cfg)
    assert decision.prediction_source == "analytical"
    assert decision.inference_latency_ms >= 0.0


def test_agent_prefers_broadcast_for_a_big_group_on_a_free_carrier():
    cfg = SimConfig.default()
    agent = AIAgent(cfg, model_dir="does-not-exist")
    cell = make_cell()
    decision = agent.decide("s0", make_group(40, cell), cell, cfg)
    assert decision.mode == DeliveryMode.BROADCAST


def test_agent_prefers_unicast_for_a_single_ue():
    """One listener gets no benefit from a shared copy."""
    cfg = SimConfig.default()
    agent = AIAgent(cfg, model_dir="does-not-exist")
    cell = make_cell()
    decision = agent.decide("s0", make_group(1, cell), cell, cfg)
    assert decision.mode == DeliveryMode.UNICAST


def test_agent_decision_is_explainable():
    cfg = SimConfig.default()
    agent = AIAgent(cfg, model_dir="does-not-exist")
    cell = make_cell()
    decision = agent.decide("s0", make_group(20, cell), cell, cfg)

    assert 0.0 <= decision.confidence <= 1.0
    assert "action" in decision.reason
    assert "winning_terms" in decision.reason
    assert decision.reason["candidate"] in ("unicast", "broadcast")


def _starved_broadcast_cell() -> Cell:
    """Same cell id, but a broadcast carrier too small to be worth using."""
    return make_cell(broadcast_capacity=0.2)


def test_cooldown_blocks_an_immediate_second_switch():
    cfg = SimConfig.default().with_overrides(cooldown_period_s=100.0, hysteresis_margin=0.0)
    agent = AIAgent(cfg, model_dir="does-not-exist")

    # 1. Start on unicast: the broadcast carrier is useless here.
    starved = _starved_broadcast_cell()
    first = agent.decide("s0", make_group(1, starved), starved, cfg)
    assert first.mode == DeliveryMode.UNICAST

    # 2. A crowded cell with a healthy carrier flips it to broadcast.
    roomy = make_cell()
    second = agent.decide("s0", make_group(80, roomy), roomy, cfg)
    assert second.mode == DeliveryMode.BROADCAST
    assert second.switched

    # 3. Conditions now favour unicast again, but cooldown must hold.
    starved_again = _starved_broadcast_cell()
    third = agent.decide("s0", make_group(1, starved_again), starved_again, cfg)
    assert third.reason["action"] == "hold_cooldown"
    assert third.mode == DeliveryMode.BROADCAST


def test_hysteresis_holds_the_current_mode_on_a_near_tie():
    cfg = SimConfig.default().with_overrides(hysteresis_margin=10.0, cooldown_period_s=0.0)
    agent = AIAgent(cfg, model_dir="does-not-exist")

    starved = _starved_broadcast_cell()
    first = agent.decide("s0", make_group(1, starved), starved, cfg)

    roomy = make_cell()
    second = agent.decide("s0", make_group(80, roomy), roomy, cfg)

    # An enormous hysteresis margin makes every gap "too small to bother".
    assert second.reason["action"] == "hold_within_hysteresis"
    assert second.mode == first.mode


def test_agent_escapes_a_mode_that_is_starving_playback():
    """Safety gates must not trap the agent in a failing mode.

    Once buffers hit zero every mode looks unsafe, so a naive buffer gate
    would block the only way out. The escape path exists for exactly this.
    """
    cfg = SimConfig.default().with_overrides(cooldown_period_s=0.0)
    agent = AIAgent(cfg, model_dir="does-not-exist")

    roomy = make_cell()
    agent.decide("s0", make_group(80, roomy), roomy, cfg)  # settles on broadcast

    # Carrier collapses AND buffers are empty - the trap condition.
    starved = _starved_broadcast_cell()
    decision = agent.decide("s0", make_group(2, starved, buffer_s=0.0), starved, cfg)

    assert decision.reason["action"] == "switch_escape_stall"
    assert decision.mode == DeliveryMode.UNICAST


def test_agent_reports_tick_stats_for_the_kpi_engine():
    cfg = SimConfig.default()
    agent = AIAgent(cfg, model_dir="does-not-exist")
    cell = make_cell()
    agent.decide("s0", make_group(10, cell), cell, cfg)

    stats = agent.current_tick_stats()
    assert stats.switch_count >= 0
    assert stats.mean_inference_latency_ms >= 0.0
    assert len(stats.decisions) == 1
