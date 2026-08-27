"""Cell-level broadcast admission: ranking, capacity feasibility, fallback."""

import pytest

from config.config import SimConfig
from traffic_sim.ai.cell_controller import GroupPlan, admit_to_broadcast, rank_by_broadcast_benefit
from traffic_sim.ai.feature_engineering import extract_features
from traffic_sim.ai.outcomes import evaluate_modes
from traffic_sim.ai.utility import compute_utility, resolve_weights
from traffic_sim.content.segment import Segment
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import DeliveryMode, UE


def make_cell(broadcast_capacity: float = 60.0) -> Cell:
    return Cell(
        cell_id="c0",
        center_x_km=0.0,
        center_y_km=0.0,
        maximum_capacity_mbps=100.0,
        broadcast_capacity_mbps=broadcast_capacity,
    )


def make_group(n: int, cell: Cell) -> list[UE]:
    ues = []
    for i in range(n):
        ue = UE(
            ue_id=f"ue_{i}",
            x_km=cell.center_x_km + 0.02 * (i % 5),
            y_km=cell.center_y_km,
            cell_id=cell.cell_id,
            buffer_seconds=10.0,
            mobility_risk=0.0,
        )
        cell.connect_ue(ue.ue_id)
        ues.append(ue)
    return ues


def make_plan(
    key: tuple[str, str],
    cell: Cell,
    n_ues: int,
    bitrate_mbps: float,
    current_mode: DeliveryMode | None = None,
    config: SimConfig | None = None,
) -> GroupPlan:
    """Build a real GroupPlan the same way AIAgent does, so these tests
    exercise the actual feature/outcome/utility pipeline, not a mock."""
    cfg = config or SimConfig.default()
    segment = Segment(segment_id=key[1], size_mb=1.0, duration_sec=10.0, bitrate_mbps=bitrate_mbps, popularity=0.5)
    ues = make_group(n_ues, cell)
    features = extract_features(segment, ues, cell, cfg)
    outcomes = evaluate_modes(features, cfg)
    weights = resolve_weights(features, cfg)
    u_unicast = compute_utility(DeliveryMode.UNICAST, outcomes.unicast, features, cfg, current_mode, weights)
    u_broadcast = compute_utility(DeliveryMode.BROADCAST, outcomes.broadcast, features, cfg, current_mode, weights)
    return GroupPlan(
        key=key,
        features=features,
        outcomes=outcomes,
        utility_unicast=u_unicast,
        utility_broadcast=u_broadcast,
        broadcast_benefit=max(0.0, u_broadcast.total - u_unicast.total),
        prediction_source="analytical",
        inference_latency_ms=0.0,
        current_mode=current_mode,
    )


# --- ranking -----------------------------------------------------------------


def test_ranking_orders_by_broadcast_benefit_descending():
    cell = make_cell(broadcast_capacity=60.0)
    # A big group gains a lot from broadcast; a single UE gains nothing.
    big = make_plan(("c0", "big"), cell, n_ues=60, bitrate_mbps=1.0)
    tiny = make_plan(("c0", "tiny"), cell, n_ues=1, bitrate_mbps=1.0)

    ranked = rank_by_broadcast_benefit([tiny, big])

    assert ranked[0].key == big.key
    assert ranked[0].broadcast_benefit >= ranked[1].broadcast_benefit


def test_ranking_is_stable_and_deterministic_for_equal_benefit():
    cell = make_cell()
    a = make_plan(("c0", "a"), cell, n_ues=10, bitrate_mbps=1.0)
    b = make_plan(("c0", "b"), cell, n_ues=10, bitrate_mbps=1.0)
    # Force an exact tie so we're testing the tie-break, not the math.
    b.broadcast_benefit = a.broadcast_benefit

    ranked_once = [p.key for p in rank_by_broadcast_benefit([a, b])]
    ranked_again = [p.key for p in rank_by_broadcast_benefit([a, b])]
    assert ranked_once == ranked_again


def test_stickiness_favours_the_currently_broadcasting_group_on_a_near_tie():
    cell = make_cell()
    incumbent = make_plan(("c0", "incumbent"), cell, n_ues=10, bitrate_mbps=1.0, current_mode=DeliveryMode.BROADCAST)
    newcomer = make_plan(("c0", "newcomer"), cell, n_ues=10, bitrate_mbps=1.0, current_mode=None)
    # Newcomer only marginally ahead - within the stickiness margin.
    newcomer.broadcast_benefit = incumbent.broadcast_benefit + 0.01

    ranked = rank_by_broadcast_benefit([incumbent, newcomer], stickiness=0.05)
    assert ranked[0].key == incumbent.key

    # A genuinely, clearly better newcomer still wins despite stickiness.
    newcomer.broadcast_benefit = incumbent.broadcast_benefit + 10.0
    ranked = rank_by_broadcast_benefit([incumbent, newcomer], stickiness=0.05)
    assert ranked[0].key == newcomer.key


# --- admission / shared-capacity feasibility ----------------------------------


def test_admission_respects_shared_broadcast_capacity():
    """Two groups whose bitrates together exceed the carrier, alone each fit."""
    cell = make_cell(broadcast_capacity=2.0)
    # Each needs ~1.5 Mbps; sharing a 2 Mbps carrier two ways gives 1.0 each
    # - not enough for either, so at most one can be admitted.
    a = make_plan(("c0", "a"), cell, n_ues=20, bitrate_mbps=1.5)
    b = make_plan(("c0", "b"), cell, n_ues=20, bitrate_mbps=1.5)

    admitted = admit_to_broadcast([a, b])

    assert len(admitted) <= 1


def test_admission_fits_more_groups_when_capacity_is_ample():
    cell = make_cell(broadcast_capacity=60.0)
    a = make_plan(("c0", "a"), cell, n_ues=20, bitrate_mbps=1.0)
    b = make_plan(("c0", "b"), cell, n_ues=20, bitrate_mbps=1.0)

    admitted = admit_to_broadcast([a, b])
    assert admitted == {a.key, b.key}


def test_lower_bitrate_candidate_can_still_fit_after_a_higher_benefit_one_is_skipped():
    """Skipping an unsustainable candidate must not block the whole list -
    a later, smaller candidate can still take the remaining headroom."""
    cell = make_cell(broadcast_capacity=2.0)
    hungry = make_plan(("c0", "hungry"), cell, n_ues=50, bitrate_mbps=5.0)  # will never fit alone
    modest = make_plan(("c0", "modest"), cell, n_ues=5, bitrate_mbps=1.0)  # fits comfortably

    admitted = admit_to_broadcast([hungry, modest])

    assert hungry.key not in admitted
    assert modest.key in admitted


def test_zero_benefit_groups_are_never_admitted():
    cell = make_cell(broadcast_capacity=60.0)
    single = make_plan(("c0", "single"), cell, n_ues=1, bitrate_mbps=1.0)  # no popularity benefit
    assert single.broadcast_benefit == 0.0

    admitted = admit_to_broadcast([single])
    assert admitted == set()


# --- non-admitted groups fall back to unicast, and consistency ---------------


def test_non_admitted_groups_are_assigned_unicast_by_the_agent():
    from traffic_sim.ai.agent import AIAgent

    cfg = SimConfig.default()
    agent = AIAgent(cfg, model_dir="does-not-exist")
    cell = make_cell(broadcast_capacity=2.0)

    groups = {
        ("c0", "a"): make_group(20, cell),
        ("c0", "b"): make_group(20, cell),
    }
    # Same cell object for both groups, as the simulator would provide.
    cells_by_id = {"c0": cell}
    # Overwrite bitrate on the segments the agent will look up for itself:
    # AIAgent has no content server here, so it falls back to a mid-range
    # bitrate assumption for every segment - fine, both groups get the
    # same demanding bitrate, so the capacity fight is still meaningful.
    agent.prepare_tick(groups, cells_by_id, cfg)

    modes = {key: agent._coordinated_modes[key] for key in groups}
    assert DeliveryMode.UNICAST in modes.values()
    # Whatever wasn't admitted to broadcast must be unicast - no third option.
    assert set(modes.values()) <= {DeliveryMode.UNICAST, DeliveryMode.BROADCAST}


def test_same_group_gets_one_consistent_decision_not_independent_contradictions():
    """The scenario this whole feature exists to fix: two big groups in one
    cell, a carrier that can only sustain one of them. Under the old
    independent-per-group logic both could see 'broadcast looks good' and
    both would pick it, oversubscribing the carrier. Coordination must
    pick at most one."""
    from traffic_sim.ai.agent import AIAgent

    cfg = SimConfig.default()
    agent = AIAgent(cfg, model_dir="does-not-exist")
    cell = make_cell(broadcast_capacity=3.0)

    group_a = make_group(40, cell)
    group_b = make_group(40, cell)
    groups = {("c0", "a"): group_a, ("c0", "b"): group_b}
    cells_by_id = {"c0": cell}

    agent.prepare_tick(groups, cells_by_id, cfg)
    mode_a = agent.decide("a", group_a, cell, cfg).mode
    mode_b = agent.decide("b", group_b, cell, cfg).mode

    assert not (mode_a == DeliveryMode.BROADCAST and mode_b == DeliveryMode.BROADCAST)


def test_decide_without_prepare_tick_still_works_standalone():
    """decide() must not crash or misbehave if prepare_tick() was never
    called - the pre-coordination fallback path."""
    from traffic_sim.ai.agent import AIAgent

    cfg = SimConfig.default()
    agent = AIAgent(cfg, model_dir="does-not-exist")
    cell = make_cell()
    decision = agent.decide("s0", make_group(5, cell), cell, cfg)
    assert decision.reason["coordination"] == "independent"
