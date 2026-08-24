"""Step 20 deliverable: run the simulator without AI and verify that
unicast and broadcast produce sensible, measurably different outcomes.

Two matched (same seeded config) runs are compared per scenario:
always-unicast vs always-broadcast. A flash-crowd scenario should show
broadcast winning on resource use/congestion; a dispersed scenario should
show no such broadcast advantage — proving the simulator genuinely models
the unicast/broadcast tradeoff rather than hardcoding a winner.
"""

import math

from config.config import SimConfig
from traffic_sim.simulation.mode_selectors import always_broadcast, always_unicast
from traffic_sim.simulation.simulator import Simulator


def run_matched(scenario: str, mode_selector, ticks: int = 60, seed: int = 123, num_ues: int = 100):
    cfg = SimConfig.default().with_overrides(random_seed=seed, num_ues=num_ues, num_cells=3, num_segments=10)
    sim = Simulator(cfg, mode_selector, scenario=scenario)
    sim.run(duration_s=ticks * cfg.simulation_timestep_s)
    return sim.kpi_engine.summary()


def test_flash_crowd_broadcast_uses_less_resource_and_congests_less():
    unicast_summary = run_matched("flash_crowd", always_unicast)
    broadcast_summary = run_matched("flash_crowd", always_broadcast)

    assert broadcast_summary["total_required_resource_mbps"] < unicast_summary["total_required_resource_mbps"]
    assert broadcast_summary["mean_congestion"] < unicast_summary["mean_congestion"]


def test_flash_crowd_broadcast_keeps_buffers_at_least_as_healthy():
    unicast_summary = run_matched("flash_crowd", always_unicast)
    broadcast_summary = run_matched("flash_crowd", always_broadcast)

    # Congestion-limited unicast throughput should not leave UEs better
    # buffered than the shared broadcast stream in a genuinely crowded cell.
    assert broadcast_summary["mean_buffer_seconds"] >= unicast_summary["mean_buffer_seconds"] - 1e-6


def test_flash_crowd_shows_a_much_larger_broadcast_advantage_than_a_sparse_scenario():
    """Broadcast's resource advantage should scale with how concentrated
    demand is. This is checked in aggregate here; the unambiguous,
    deterministic proof that broadcast has NO advantage at all when
    exactly one UE requests a segment (R_unicast == R_broadcast, a pure
    tie) lives in tests/test_delivery_unicast_broadcast.py — full-population
    statistical runs are too noisy for that exact boundary case because
    content-popularity skew alone (independent of the flash-crowd effect)
    already creates multi-UE groups."""
    flash_summary = run_matched("flash_crowd", always_broadcast, seed=999)
    # A tiny, deliberately sparse population (few UEs across many
    # segments/cells) keeps most (cell, segment) groups near size 1, so
    # broadcast's flat per-group cost buys much less relative to demand
    # than the flash-crowd's single large concentrated group.
    sparse_summary = run_matched("uniform", always_broadcast, seed=999, num_ues=6)

    assert flash_summary["resource_savings_vs_baseline"] > sparse_summary["resource_savings_vs_baseline"]


def test_simulation_runs_long_without_nan_or_exceptions():
    cfg = SimConfig.default().with_overrides(random_seed=7, num_ues=60, num_cells=3, num_segments=10)
    sim = Simulator(cfg, always_broadcast, scenario="flash_crowd")
    sim.run(duration_s=300)

    for record in sim.kpi_engine.to_records():
        for key, value in record.items():
            if isinstance(value, float):
                assert not math.isnan(value), f"{key} is NaN"
                assert not math.isinf(value) or key in ("mean_spectrum_efficiency",), f"{key} is inf"

    for ue in sim.ues:
        assert ue.buffer_seconds >= 0.0
        assert not math.isnan(ue.buffer_seconds)
