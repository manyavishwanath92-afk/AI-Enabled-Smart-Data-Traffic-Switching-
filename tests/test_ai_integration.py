"""The AI layer running inside the real simulator, end to end."""

import pytest

from config.config import SimConfig
from traffic_sim.ai.dataset import LABEL_NAMES, generate_dataset
from traffic_sim.ai.feature_engineering import FEATURE_NAMES
from traffic_sim.experiments.compare import run_comparison
from traffic_sim.main import build_simulator


def run_ai(scenario="flash_crowd", ticks=30, seed=11, **overrides):
    cfg = SimConfig.default().with_overrides(random_seed=seed, **overrides)
    simulator = build_simulator(cfg, "ai", scenario)
    simulator.run(duration_s=ticks * cfg.simulation_timestep_s)
    return simulator


def mode_share(simulator) -> dict[str, float]:
    counts: dict[str, int] = {}
    for record in simulator.kpi_engine.to_records():
        for part in record["mode_breakdown"].split(","):
            if "=" in part:
                mode, value = part.split("=")
                counts[mode] = counts.get(mode, 0) + int(value)
    total = sum(counts.values()) or 1
    return {mode: count / total for mode, count in counts.items()}


def test_ai_selector_runs_inside_the_simulator():
    simulator = run_ai()
    summary = simulator.kpi_engine.summary()

    assert summary["num_ticks"] == 30
    assert summary["service_continuity"] > 0.0


def test_ai_agent_is_wired_to_the_content_catalogue():
    simulator = run_ai()
    assert simulator.mode_selector._content_server is simulator.content_server


def test_mbos_and_pcf_run_on_every_real_decision():
    """End-to-end proof that MBOS/PCF are actually wired into the live
    pipeline, not just unit-testable in isolation.

    `current_tick_stats()` is only populated between the end of a tick's
    decisions and the following `observe()` call, so decisions are
    captured live via a spy rather than read after the run completes
    (by then the last tick's `observe()` has already cleared them).
    """
    from traffic_sim.ai.agent import AIAgent

    cfg = SimConfig.default().with_overrides(random_seed=11)
    simulator = build_simulator(cfg, "ai", "flash_crowd")
    captured = []
    original_decide = AIAgent.decide

    def spy(self, *args, **kwargs):
        decision = original_decide(self, *args, **kwargs)
        captured.append(decision)
        return decision

    AIAgent.decide = spy
    try:
        simulator.run(duration_s=10 * cfg.simulation_timestep_s)
    finally:
        AIAgent.decide = original_decide

    assert len(captured) > 0
    for decision in captured:
        assert decision.session_id  # non-empty "cell_id:segment_id"
        assert decision.target_bearer in ("cellular", "broadcast")
        assert isinstance(decision.handover, bool)
        assert isinstance(decision.pcf_compliant, bool)
        assert isinstance(decision.pcf_violated_rules, list)
        assert "mbos" in decision.reason and "pcf" in decision.reason


def test_baselines_never_touch_mbos_or_pcf():
    """MBOS/PCF are part of the AI pipeline specifically - fixed-rule
    selectors must remain raw, unpoliced comparison points."""
    from traffic_sim.simulation.mode_selectors import always_broadcast

    cfg = SimConfig.default().with_overrides(random_seed=11)
    simulator = build_simulator(cfg, "always-broadcast", "flash_crowd")
    assert simulator.mode_selector is always_broadcast
    assert not hasattr(simulator.mode_selector, "mbos")
    assert not hasattr(simulator.mode_selector, "pcf")
    assert not hasattr(simulator.mode_selector, "decision_history")


# --- decision history (dashboard/API telemetry) -----------------------------


def _agent_and_group(**overrides):
    from traffic_sim.ai.agent import AIAgent
    from traffic_sim.simulation.cell import Cell
    from traffic_sim.simulation.user import UE

    cfg = SimConfig.default().with_overrides(**overrides) if overrides else SimConfig.default()
    agent = AIAgent(cfg)
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=100.0)
    ues = [UE(ue_id="u0", x_km=0.1, y_km=0.0, cell_id="c0", buffer_seconds=10.0)]
    return cfg, agent, cell, ues


def test_decision_history_selected_mode_is_final_after_pcf_reverts():
    """The history must record what PCF actually allowed, not the agent's
    pre-PCF candidate - proven by forcing PCF to revert a decision and
    checking the persisted DecisionRecord reflects the reversion."""
    from types import SimpleNamespace

    from traffic_sim.policy.pcf import PRIORITY_BUFFER_MARGIN, PolicyEvaluation
    from traffic_sim.simulation.user import DeliveryMode

    cfg, agent, cell, ues = _agent_and_group()

    def fake_enforce(mode, current_mode, service_priority, buffer_min_s, confidence):
        # Whatever MBOS proposed, PCF reverts it to UNICAST.
        return PolicyEvaluation(
            final_mode=DeliveryMode.UNICAST, compliant=False, violated_rules=[PRIORITY_BUFFER_MARGIN]
        )

    agent.pcf.enforce = fake_enforce
    decision = agent.decide("seg_0", ues, cell, cfg)
    assert decision.mode == DeliveryMode.UNICAST  # AIDecision itself already reconciled

    agent.observe([cell], [], SimpleNamespace(tick=1, sim_time_s=1.0))
    history = agent.decision_history()

    assert len(history) == 1
    assert history[0].selected_mode == "unicast"
    assert history[0].pcf_result == f"reverted:{PRIORITY_BUFFER_MARGIN}"


def test_decision_history_previous_mode_matches_prior_selected_mode():
    """Across two ticks for the same group, the second row's previous_mode
    must equal the first row's selected_mode - proving continuity is
    tracked correctly regardless of which mode was actually chosen."""
    from types import SimpleNamespace

    cfg, agent, cell, ues = _agent_and_group()

    for tick in (1, 2):
        agent.decide("seg_0", ues, cell, cfg)
        agent.observe([cell], [], SimpleNamespace(tick=tick, sim_time_s=tick * cfg.simulation_timestep_s))

    history = agent.decision_history()
    assert len(history) == 2
    assert history[0].previous_mode is None  # first-ever decision for this group
    assert history[1].previous_mode == history[0].selected_mode


def test_decision_history_mbos_bearer_and_pcf_result_reflect_final_mode():
    from types import SimpleNamespace

    from traffic_sim.mbos.controller import bearer_for
    from traffic_sim.simulation.user import DeliveryMode

    cfg, agent, cell, ues = _agent_and_group()
    agent.decide("seg_0", ues, cell, cfg)
    agent.observe([cell], [], SimpleNamespace(tick=1, sim_time_s=1.0))

    record = agent.decision_history()[0]
    assert record.mbos_bearer == bearer_for(DeliveryMode(record.selected_mode))
    assert record.pcf_result == "compliant"  # nothing forces a violation in this scenario


def test_decision_history_is_bounded_to_the_most_recent_records():
    from types import SimpleNamespace

    cfg, agent, cell, ues = _agent_and_group(decision_history_size=3)

    for tick in range(1, 8):  # far more ticks than the configured cap
        agent.decide("seg_0", ues, cell, cfg)
        agent.observe([cell], [], SimpleNamespace(tick=tick, sim_time_s=float(tick)))

    history = agent.decision_history()
    assert len(history) == 3
    assert [r.tick for r in history] == [5, 6, 7]  # oldest dropped, newest kept


def test_decision_history_populated_by_a_real_simulator_run():
    """End-to-end: a real simulator run must leave decision_history
    populated and survivable past the run, unlike current_tick_stats()
    which is cleared by the final tick's observe()."""
    simulator = run_ai(ticks=15)
    history = simulator.mode_selector.decision_history()

    assert len(history) > 0
    kpi_ticks = {r["tick"] for r in simulator.kpi_engine.to_records()}
    assert {r.tick for r in history} <= kpi_ticks
    for r in history:
        assert r.selected_mode in ("unicast", "broadcast")
        assert r.mbos_bearer in ("cellular", "broadcast")


def test_switch_count_and_inference_latency_are_no_longer_placeholders():
    simulator = run_ai()
    records = simulator.kpi_engine.to_records()

    assert all(r["switch_count"] is not None for r in records)
    assert all(r["ai_inference_latency_ms"] is not None for r in records)
    assert simulator.kpi_engine.summary()["mean_inference_latency_ms"] > 0.0


def test_ai_meets_the_decision_latency_requirement():
    """Inference must fit comfortably inside one segment interval."""
    simulator = run_ai()
    latency_ms = simulator.kpi_engine.summary()["mean_inference_latency_ms"]
    assert latency_ms < 1000.0 * simulator.config.simulation_timestep_s


def test_ai_offloads_a_flash_crowd_to_broadcast_when_the_carrier_is_free():
    simulator = run_ai(broadcast_cell_capacity_mbps=60.0)
    assert mode_share(simulator).get("broadcast", 0.0) > 0.5


def test_ai_backs_off_broadcast_when_the_carrier_is_starved():
    """The decision must depend on conditions, not be a constant.

    This is the difference between an agent and a hardcoded rule: the
    same code, the same traffic, a smaller broadcast carrier, and a
    materially different policy comes out.
    """
    free = mode_share(run_ai(broadcast_cell_capacity_mbps=60.0)).get("broadcast", 0.0)
    starved = mode_share(run_ai(broadcast_cell_capacity_mbps=4.0)).get("broadcast", 0.0)

    assert starved < free - 0.2


def test_ai_protects_continuity_relative_to_blind_broadcasting():
    """With a starved carrier, always-broadcast wrecks playback. The AI
    declining to broadcast everything must measurably beat it."""
    cfg = SimConfig.default().with_overrides(broadcast_cell_capacity_mbps=4.0)
    rows = {row.policy: row.summary for row in run_comparison(cfg, ticks=30, seeds=(11, 22))}

    assert rows["ai"]["service_continuity"] > rows["always-broadcast"]["service_continuity"]
    assert rows["ai"]["mean_qoe"] > rows["always-broadcast"]["mean_qoe"]


def test_cell_level_coordination_sharply_reduces_switching_under_a_constrained_carrier():
    """Before cell-level admission control, every group reacted to the
    same shared-carrier signal independently and the whole cell herded
    on and off broadcast together - measured at ~99 switches/60 ticks
    under a 4 Mbps carrier. Coordinating the decision once per cell must
    bring that down decisively, not just marginally."""
    cfg = SimConfig.default().with_overrides(broadcast_cell_capacity_mbps=4.0)
    rows = {row.policy: row.summary for row in run_comparison(cfg, ticks=60, seeds=(11, 22, 33))}

    assert rows["ai"]["total_switches"] < 60.0  # well below the ~99 measured before coordination


def test_ai_beats_the_naive_rule_on_continuity_under_a_constrained_carrier():
    """The naive group-size rule doesn't know the carrier is contended at
    all. Once coordination stops the AI from oscillating, it should
    out-perform a rule that can't reason about shared capacity."""
    cfg = SimConfig.default().with_overrides(broadcast_cell_capacity_mbps=4.0)
    rows = {row.policy: row.summary for row in run_comparison(cfg, ticks=60, seeds=(11, 22, 33))}

    assert rows["ai"]["service_continuity"] > rows["rule-by-group-size"]["service_continuity"]


def test_ai_saves_resources_relative_to_always_unicast_when_broadcast_is_viable():
    cfg = SimConfig.default().with_overrides(broadcast_cell_capacity_mbps=60.0)
    rows = {row.policy: row.summary for row in run_comparison(cfg, ticks=30, seeds=(11, 22))}

    assert rows["ai"]["resource_savings_vs_baseline"] > 0.5
    assert rows["ai"]["service_continuity"] >= rows["always-unicast"]["service_continuity"]


# --- dataset and model round trip -------------------------------------------


def test_dataset_has_the_expected_columns_and_sane_labels():
    dataset = generate_dataset(seeds=(1,), scenarios=("flash_crowd",), ticks=3)

    assert len(dataset) > 0
    for row in dataset.rows:
        assert set(row) == set(FEATURE_NAMES) | set(LABEL_NAMES)
        assert row["y_unicast_mbps"] >= 0.0
        assert row["y_broadcast_mbps"] >= 0.0


def test_trained_model_round_trips_and_beats_a_naive_guess(tmp_path):
    """Train small, save, reload, and check it actually learned something."""
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")

    from traffic_sim.ai.predict import OutcomePredictor
    from traffic_sim.ai.train import train_models

    dataset = generate_dataset(seeds=(1, 2), scenarios=("flash_crowd", "uniform"), ticks=6)
    report = train_models(dataset, tmp_path)

    for mode in ("unicast", "broadcast"):
        assert report["modes"][mode]["r2"] > 0.5, f"{mode} model did not learn"

    # The saved artifacts must load back into a working predictor.
    cfg = SimConfig.default()
    predictor = OutcomePredictor(cfg, model_dir=tmp_path)
    assert predictor.uses_model is True


def test_predictor_reports_which_path_it_used(tmp_path):
    """Analytical results must never be silently labelled as model output."""
    from traffic_sim.ai.predict import OutcomePredictor
    from traffic_sim.ai.feature_engineering import extract_features
    from traffic_sim.simulation.cell import Cell
    from traffic_sim.content.segment import Segment
    from traffic_sim.simulation.user import UE

    cfg = SimConfig.default()
    predictor = OutcomePredictor(cfg, model_dir=tmp_path / "empty")
    assert predictor.uses_model is False

    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=100.0)
    ue = UE(ue_id="u", x_km=0.1, y_km=0.0, cell_id="c0", buffer_seconds=5.0)
    segment = Segment(segment_id="s0", size_mb=2.5, duration_sec=10.0, bitrate_mbps=2.0)
    features = extract_features(segment, [ue], cell, cfg)

    assert predictor.predict(features).source == "analytical"
