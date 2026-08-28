"""The dashboard API serving real simulator output.

These tests exercise the HTTP/WebSocket layer against genuinely running
simulations — nothing is stubbed, so a passing suite means the endpoints Manus
will call return real data in the documented shape.
"""

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from config.config import SimConfig  # noqa: E402
from traffic_sim.api.server import create_app  # noqa: E402

# Wall-clock measurements legitimately vary between runs (a background thread
# is not the main thread); every other KPI is deterministic given the seed.
WALL_CLOCK_KPIS = {"mean_inference_latency_ms"}


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def start_run(client, ticks=8, mode_selector="ai", scenario="flash_crowd", seed=11, **extra):
    body = {
        "scenario": scenario,
        "mode_selector": mode_selector,
        "ticks": ticks,
        "seed": seed,
        "tick_interval_s": 0.0,
        **extra,
    }
    response = client.post("/api/simulations", json=body)
    assert response.status_code == 201, response.text
    return response.json()["simulation_id"]


def run_to_completion(client, simulation_id, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/simulations/{simulation_id}/status").json()
        if status["status"] != "running":
            return status
        time.sleep(0.01)
    raise AssertionError(f"simulation {simulation_id} did not finish within {timeout}s")


# --- meta -------------------------------------------------------------------


def test_health_reports_ok(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_config_endpoint_returns_the_real_simconfig(client):
    payload = client.get("/api/config").json()
    expected = SimConfig.default().to_dict()

    # Every SimConfig field must be served, with matching values — the
    # dashboard reads its capacity/threshold labels straight from here.
    assert set(payload) == set(expected)
    for key, value in expected.items():
        assert payload[key] == value, key


def test_scenarios_endpoint_lists_valid_control_values(client):
    payload = client.get("/api/scenarios").json()
    assert "flash_crowd" in payload["scenarios"]
    assert "ai" in payload["mode_selectors"]
    assert "always-broadcast" in payload["mode_selectors"]


# --- lifecycle --------------------------------------------------------------


def test_simulation_runs_to_completion_and_reports_progress(client):
    simulation_id = start_run(client, ticks=8)
    status = run_to_completion(client, simulation_id)

    assert status["status"] == "completed"
    assert status["current_tick"] == 8
    assert status["total_ticks"] == 8
    assert status["error"] is None


def test_simulations_can_be_listed_stopped_and_deleted(client):
    simulation_id = start_run(client, ticks=500, tick_interval_s=0.05)

    listed = client.get("/api/simulations").json()
    assert any(row["simulation_id"] == simulation_id for row in listed)

    stopped = client.post(f"/api/simulations/{simulation_id}/stop").json()
    assert stopped["status"] == "stopped"
    assert stopped["current_tick"] < 500  # genuinely interrupted, not run out

    assert client.delete(f"/api/simulations/{simulation_id}").json()["deleted"] is True
    assert client.get(f"/api/simulations/{simulation_id}/status").status_code == 404


def test_simulations_can_be_paused_and_resumed(client):
    simulation_id = start_run(client, ticks=500, tick_interval_s=0.02)

    paused = client.post(f"/api/simulations/{simulation_id}/pause").json()
    assert paused["status"] == "paused"

    # A pause must actually stop the clock, not just relabel the run.
    tick_at_pause = client.get(f"/api/simulations/{simulation_id}/status").json()["current_tick"]
    time.sleep(0.3)
    assert client.get(f"/api/simulations/{simulation_id}/status").json()["current_tick"] == tick_at_pause

    assert client.post(f"/api/simulations/{simulation_id}/resume").json()["status"] == "running"
    time.sleep(0.3)
    assert client.get(f"/api/simulations/{simulation_id}/status").json()["current_tick"] > tick_at_pause

    client.post(f"/api/simulations/{simulation_id}/stop")


def test_pause_and_resume_reject_nonsensical_transitions(client):
    simulation_id = start_run(client, ticks=500, tick_interval_s=0.02)

    # Not paused yet, so there is nothing to resume.
    assert client.post(f"/api/simulations/{simulation_id}/resume").status_code == 409

    client.post(f"/api/simulations/{simulation_id}/pause")
    assert client.post(f"/api/simulations/{simulation_id}/pause").status_code == 409

    client.post(f"/api/simulations/{simulation_id}/stop")


def test_a_paused_simulation_can_still_be_stopped(client):
    """Otherwise a paused run would wait forever and never release."""
    simulation_id = start_run(client, ticks=500, tick_interval_s=0.02)
    client.post(f"/api/simulations/{simulation_id}/pause")

    assert client.post(f"/api/simulations/{simulation_id}/stop").json()["status"] == "stopped"


def test_unknown_simulation_id_is_a_404(client):
    assert client.get("/api/simulations/sim_nope/status").status_code == 404
    assert client.get("/api/simulations/sim_nope/kpis/latest").status_code == 404


def test_invalid_selector_and_scenario_are_rejected(client):
    bad_selector = client.post("/api/simulations", json={"mode_selector": "telepathy", "ticks": 2})
    assert bad_selector.status_code == 400

    bad_scenario = client.post("/api/simulations", json={"scenario": "moon_base", "ticks": 2})
    assert bad_scenario.status_code == 400

    bad_ticks = client.post("/api/simulations", json={"ticks": 0})
    assert bad_ticks.status_code == 422  # pydantic bounds


def test_config_overrides_reach_the_simulation(client):
    """The dashboard can demo a starved carrier without a backend change."""
    simulation_id = start_run(client, ticks=4, config_overrides={"broadcast_cell_capacity_mbps": 4.0})
    run_to_completion(client, simulation_id)

    cells = client.get(f"/api/simulations/{simulation_id}/cells").json()
    assert all(c["broadcast_capacity_mbps"] == 4.0 for c in cells)


def test_unknown_config_override_is_rejected(client):
    response = client.post(
        "/api/simulations", json={"ticks": 2, "config_overrides": {"warp_factor": 9}}
    )
    assert response.status_code == 400
    assert "warp_factor" in response.json()["detail"]


# --- KPIs -------------------------------------------------------------------


def test_kpi_latest_history_and_summary_come_from_the_kpi_engine(client):
    simulation_id = start_run(client, ticks=10)
    run_to_completion(client, simulation_id)

    latest = client.get(f"/api/simulations/{simulation_id}/kpis/latest").json()
    assert latest["tick"] == 10
    assert 0.0 <= latest["service_continuity"] <= 1.0
    assert "mode_breakdown" in latest

    history = client.get(f"/api/simulations/{simulation_id}/kpis/history").json()
    assert len(history) == 10
    assert [r["tick"] for r in history] == list(range(1, 11))
    assert history[-1] == latest

    summary = client.get(f"/api/simulations/{simulation_id}/kpis/summary").json()
    assert summary["num_ticks"] == 10
    assert "resource_savings_vs_baseline" in summary


def test_kpi_history_respects_the_tick_range(client):
    simulation_id = start_run(client, ticks=10)
    run_to_completion(client, simulation_id)

    window = client.get(
        f"/api/simulations/{simulation_id}/kpis/history", params={"from_tick": 4, "to_tick": 6}
    ).json()
    assert [r["tick"] for r in window] == [4, 5, 6]


# --- network state ----------------------------------------------------------


def test_cells_expose_live_state_including_computed_congestion(client):
    simulation_id = start_run(client, ticks=5)
    run_to_completion(client, simulation_id)

    cells = client.get(f"/api/simulations/{simulation_id}/cells").json()
    assert len(cells) == SimConfig.default().num_cells
    for cell in cells:
        assert 0.0 <= cell["congestion"] <= 1.0  # congestion() is a method, not a field
        assert cell["num_connected_ues"] == len(cell["connected_ue_ids"])


def test_ues_are_paginated_and_filterable_by_cell(client):
    simulation_id = start_run(client, ticks=5)
    run_to_completion(client, simulation_id)

    page = client.get(f"/api/simulations/{simulation_id}/ues", params={"limit": 5}).json()
    assert page["total"] == SimConfig.default().num_ues
    assert len(page["items"]) == 5
    # Enums must arrive as plain strings, not "DeliveryMode.UNICAST".
    assert page["items"][0]["delivery_mode"] in ("none", "unicast", "broadcast")
    assert page["items"][0]["qoe_state"] in ("good", "degraded", "starved")

    cell_id = client.get(f"/api/simulations/{simulation_id}/cells").json()[0]["cell_id"]
    filtered = client.get(f"/api/simulations/{simulation_id}/ues", params={"cell_id": cell_id}).json()
    assert filtered["total"] < page["total"]
    assert all(u["cell_id"] == cell_id for u in filtered["items"])


def test_groups_latest_exposes_real_per_group_delivery_detail(client):
    """DeliveryResult is discarded by the simulator each tick; the API's
    capture hook is what makes this per-group breakdown available at all."""
    simulation_id = start_run(client, ticks=5)
    run_to_completion(client, simulation_id)

    groups = client.get(f"/api/simulations/{simulation_id}/groups/latest").json()
    assert len(groups) > 0
    for group in groups:
        assert group["mode"] in ("unicast", "broadcast")
        assert group["n_requesting"] > 0
        assert group["required_resource_mbps"] >= 0.0
        assert set(group["achievable_mbps_by_ue"])  # per-UE rates really present


# --- AI decisions -----------------------------------------------------------


def test_decision_history_is_served_for_the_ai_selector(client):
    simulation_id = start_run(client, ticks=10, mode_selector="ai")
    run_to_completion(client, simulation_id)

    history = client.get(f"/api/simulations/{simulation_id}/decisions/history").json()
    assert len(history) > 0
    for row in history:
        assert row["selected_mode"] in ("unicast", "broadcast")
        assert row["mbos_bearer"] in ("cellular", "broadcast")
        assert row["pcf_result"] == "compliant" or row["pcf_result"].startswith("reverted:")
        assert 1 <= row["tick"] <= 10


def test_decision_history_limit_returns_the_most_recent_rows(client):
    simulation_id = start_run(client, ticks=10)
    run_to_completion(client, simulation_id)

    full = client.get(f"/api/simulations/{simulation_id}/decisions/history").json()
    trimmed = client.get(
        f"/api/simulations/{simulation_id}/decisions/history", params={"limit": 3}
    ).json()
    assert len(trimmed) == 3
    assert trimmed == full[-3:]


def test_decisions_latest_shows_final_mode_confidence_and_a_plain_reason(client):
    """What the dashboard's explainability panel renders: the final mode,
    how sure the agent was, and one readable sentence."""
    simulation_id = start_run(client, ticks=5)
    run_to_completion(client, simulation_id)

    decisions = client.get(f"/api/simulations/{simulation_id}/decisions/latest").json()
    assert len(decisions) > 0
    for decision in decisions:
        # The FINAL mode, after MBOS coordination and PCF enforcement.
        assert decision["selected_mode"] in ("unicast", "broadcast")
        assert decision["bearer"] in ("cellular", "broadcast")
        assert 0.0 <= decision["confidence"] <= 1.0
        assert decision["explanation"].endswith(".")
        assert decision["selected_mode"] in decision["explanation"]
        assert isinstance(decision["switched"], bool)


def test_decisions_latest_does_not_leak_the_internal_reasoning_chain(client):
    """The prediction -> utility -> agent internals stay in the backend; the
    dashboard is meant to show a conclusion, not the whole derivation."""
    simulation_id = start_run(client, ticks=5)
    run_to_completion(client, simulation_id)

    decisions = client.get(f"/api/simulations/{simulation_id}/decisions/latest").json()
    assert len(decisions) > 0
    internal = {
        "reason",
        "winning_terms",
        "utility_unicast",
        "utility_broadcast",
        "predicted_unicast_mbps",
        "predicted_broadcast_mbps",
        "prediction_source",
    }
    for decision in decisions:
        assert internal.isdisjoint(decision), f"leaked internals: {internal & set(decision)}"


def test_decision_history_rows_carry_a_readable_explanation(client):
    simulation_id = start_run(client, ticks=8)
    run_to_completion(client, simulation_id)

    history = client.get(f"/api/simulations/{simulation_id}/decisions/history").json()
    assert len(history) > 0
    for row in history:
        assert row["explanation"]
        # The sentence must describe the mode that was actually delivered.
        assert row["selected_mode"] in row["explanation"]
        # ...and never restate the raw internal code.
        assert row["decision_reason"] not in row["explanation"]


def test_baseline_selectors_expose_no_ai_decisions(client):
    """Fixed-rule baselines never run MBOS/PCF or record decisions, so the AI
    endpoints must report that honestly rather than inventing rows."""
    simulation_id = start_run(client, ticks=5, mode_selector="always-broadcast")
    run_to_completion(client, simulation_id)

    assert client.get(f"/api/simulations/{simulation_id}/decisions/history").json() == []
    assert client.get(f"/api/simulations/{simulation_id}/decisions/latest").json() == []
    # ...but the run itself still produces full KPIs.
    assert client.get(f"/api/simulations/{simulation_id}/kpis/summary").json()["num_ticks"] == 5


# --- comparison -------------------------------------------------------------


def test_comparison_returns_every_policy_row(client):
    rows = client.get(
        "/api/comparison", params={"scenario": "flash_crowd", "ticks": 5, "seeds": [11]}
    ).json()

    policies = {row["policy"] for row in rows}
    assert policies == {"always-unicast", "always-broadcast", "rule-by-group-size", "ai"}
    for row in rows:
        assert "service_continuity" in row["summary"]
        assert "resource_savings_vs_baseline" in row["summary"]


def test_comparison_rejects_an_unknown_scenario(client):
    assert client.get("/api/comparison", params={"scenario": "moon_base", "ticks": 5}).status_code == 400


# --- live updates -----------------------------------------------------------


def test_websocket_streams_one_message_per_tick_then_a_final_status(client):
    simulation_id = start_run(client, ticks=5, tick_interval_s=0.05)

    ticks = []
    with client.websocket_connect(f"/api/simulations/{simulation_id}/live") as ws:
        while True:
            message = ws.receive_json()
            if message["type"] == "tick":
                ticks.append(message["tick"])
                assert message["kpi"]["tick"] == message["tick"]
                assert isinstance(message["decisions"], list)
                assert isinstance(message["groups"], list)
                # Cells ride along so the live network view needs no polling.
                assert len(message["cells"]) == SimConfig.default().num_cells
                assert all("congestion" in c for c in message["cells"])
            else:
                assert message["type"] == "status"
                assert message["status"] == "completed"
                break

    assert ticks == [1, 2, 3, 4, 5]  # every tick delivered, in order, none dropped


def test_websocket_decisions_carry_the_final_mode_and_an_explanation(client):
    """The dashboard reads decisions off the stream, so the same guarantee
    the REST endpoint gives must hold there too."""
    simulation_id = start_run(client, ticks=4, tick_interval_s=0.05)

    seen = 0
    with client.websocket_connect(f"/api/simulations/{simulation_id}/live") as ws:
        while True:
            message = ws.receive_json()
            if message["type"] != "tick":
                break
            for decision in message["decisions"]:
                assert decision["selected_mode"] in ("unicast", "broadcast")
                assert decision["explanation"]
                assert "reason" not in decision
                seen += 1

    assert seen > 0


def test_websocket_rejects_an_unknown_simulation(client):
    with client.websocket_connect("/api/simulations/sim_nope/live") as ws:
        assert ws.receive_json()["type"] == "error"


# --- the API must not change simulation results -----------------------------


def test_api_run_matches_a_direct_simulator_run_exactly(client):
    """Same seed, scenario and tick count through the API and through
    `build_simulator()` must produce identical KPIs — proof that the serving
    layer (and its per-tick capture hook) observes without perturbing."""
    from traffic_sim.main import build_simulator

    simulation_id = start_run(client, ticks=12, mode_selector="ai", seed=11)
    run_to_completion(client, simulation_id)
    api_summary = client.get(f"/api/simulations/{simulation_id}/kpis/summary").json()

    config = SimConfig.default().with_overrides(random_seed=11)
    simulator = build_simulator(config, "ai", "flash_crowd")
    simulator.run(duration_s=12 * config.simulation_timestep_s)
    direct_summary = simulator.kpi_engine.summary()

    for key, expected in direct_summary.items():
        if key in WALL_CLOCK_KPIS:
            continue
        assert api_summary[key] == pytest.approx(expected, abs=1e-12), key


# --- explanation rendering --------------------------------------------------


def test_every_agent_reason_code_renders_a_readable_sentence():
    """Each code the agent can emit must have a sentence — a missing one
    would surface as raw jargon in the dashboard."""
    from traffic_sim.api.explain import explain_decision

    codes = [
        "initial_assignment",
        "hold_already_optimal",
        "switch",
        "switch_escape_stall",
        "hold_buffer_unsafe",
        "hold_low_confidence",
        "hold_within_hysteresis",
        "hold_cooldown",
        "hold_pcf_policy_violation",
    ]
    for code in codes:
        sentence = explain_decision(code, "broadcast", previous_mode="unicast")
        assert sentence.endswith(".")
        assert "broadcast" in sentence
        assert code not in sentence, f"{code} leaked its raw code into the text"


def test_pcf_violations_name_the_reason_in_plain_words():
    from traffic_sim.api.explain import explain_decision

    sentence = explain_decision(
        "hold_pcf_policy_violation", "unicast", "broadcast", ["priority_buffer_margin"]
    )
    assert "buffered playback" in sentence
    assert "priority_buffer_margin" not in sentence


def test_an_unknown_reason_code_degrades_instead_of_raising():
    """A serving layer must not 500 because the agent gained a new code."""
    from traffic_sim.api.explain import explain_decision

    sentence = explain_decision("some_future_code", "unicast")
    assert "unicast" in sentence
    assert sentence.endswith(".")


def test_history_rows_unpack_pcf_rules_from_their_stored_form():
    """History stores PCF's verdict as 'reverted:<rule>', not a rule list."""
    from traffic_sim.api.explain import explain_record

    sentence = explain_record(
        {
            "decision_reason": "hold_pcf_policy_violation",
            "selected_mode": "unicast",
            "previous_mode": "broadcast",
            "pcf_result": "reverted:confidence_floor",
        }
    )
    assert "confident enough" in sentence
    assert "confidence_floor" not in sentence
