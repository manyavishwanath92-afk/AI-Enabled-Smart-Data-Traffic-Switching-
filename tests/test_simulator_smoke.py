"""Step 20 deliverable: run the simulator without AI and verify that
unicast and broadcast produce sensible, measurably different outcomes.

Two matched (same seeded config) runs are compared per scenario:
always-unicast vs always-broadcast. A flash-crowd scenario should show
broadcast winning on resource use/congestion; a dispersed scenario should
show no such broadcast advantage — proving the simulator genuinely models
the unicast/broadcast tradeoff rather than hardcoding a winner.
"""

import math

import pytest

from config.config import SimConfig
from traffic_sim.simulation.mode_selectors import always_broadcast, always_unicast, rule_by_group_size
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


# --- Phase 2: radio conditions now drive delivery ---------------------------


def test_service_continuity_is_a_real_nonzero_measurement():
    """Service continuity is measured from actual playback stalls, so it
    only reads non-zero if the whole chain works: the noise-rise model
    must give usable SINR, which must give enough Shannon capacity, which
    must give an achievable rate near the segment bitrate, which must keep
    buffers off empty. A break anywhere upstream shows up here as 0.
    """
    summary = run_matched("flash_crowd", always_broadcast)
    assert summary["service_continuity"] > 0.0
    assert summary["mean_sinr_db"] > 5.0


def test_flash_crowd_unicast_starves_while_broadcast_sustains_playback():
    """Radio-driven, not accounting-driven: under a flash crowd the unicast
    UEs share the cell's capacity and starve, while the single shared
    broadcast transmission keeps every listener playing."""
    unicast_summary = run_matched("flash_crowd", always_unicast)
    broadcast_summary = run_matched("flash_crowd", always_broadcast)

    assert broadcast_summary["mean_achievable_mbps"] > unicast_summary["mean_achievable_mbps"]
    assert broadcast_summary["service_continuity"] > unicast_summary["service_continuity"]
    assert broadcast_summary["buffer_starvation_rate"] < unicast_summary["buffer_starvation_rate"]
    assert broadcast_summary["total_rebuffering_events"] < unicast_summary["total_rebuffering_events"]


def test_broadcast_offload_relieves_the_remaining_unicast_users():
    """Two-pass step(): UEs moved to broadcast stop consuming cellular
    share, so the UEs still on unicast get more throughput each. Compared
    at equal cell population, with only the offload differing."""
    cfg = SimConfig.default().with_overrides(random_seed=5, num_ues=120, num_cells=1)

    all_unicast = Simulator(cfg, always_unicast, scenario="flash_crowd")
    all_unicast.run(duration_s=30)

    # rule-by-group-size offloads the big flash-crowd group to broadcast,
    # leaving the small groups on unicast with the cell largely to themselves.
    mixed = Simulator(cfg, rule_by_group_size(min_group_size=20), scenario="flash_crowd")
    mixed.run(duration_s=30)

    assert mixed.kpi_engine.summary()["mean_achievable_mbps"] > all_unicast.kpi_engine.summary()[
        "mean_achievable_mbps"
    ]


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


# --- reproducibility --------------------------------------------------------


def test_same_seed_produces_identical_results():
    """The matched-seed premise the whole AI-vs-baseline comparison rests on.

    Two runs of the same seed and scenario must agree exactly — not merely
    closely.
    """
    summaries = []
    for _ in range(2):
        cfg = SimConfig.default().with_overrides(random_seed=11)
        sim = Simulator(cfg, always_unicast, scenario="flash_crowd")
        sim.run(duration_s=30 * cfg.simulation_timestep_s)
        summaries.append(sim.kpi_engine.summary())

    assert summaries[0] == summaries[1]


def test_same_seed_is_reproducible_across_processes():
    """Regression guard for a real bug: the flash-crowd cell was drawn from
    an unordered set, so Python's per-process string-hash randomisation put
    the crowd on a different cell for the *same* seed in different processes
    (measured spread in service continuity: ~0.78-0.86).

    This has to run in subprocesses to be meaningful. Within one process the
    hash seed is fixed, so a set of the same strings always iterates the same
    way and the bug is invisible — an in-process test passes either way.
    PYTHONHASHSEED is what varies the ordering, so the check is: same sim
    seed, different hash seeds, identical results.
    """
    import os
    import subprocess
    import sys

    script = (
        "from config.config import SimConfig;"
        "from traffic_sim.simulation.mode_selectors import always_unicast;"
        "from traffic_sim.simulation.simulator import Simulator;"
        "c = SimConfig.default().with_overrides(random_seed=11);"
        "s = Simulator(c, always_unicast, scenario='flash_crowd');"
        "s.run(duration_s=20 * c.simulation_timestep_s);"
        "print('%.12f' % s.kpi_engine.summary()['service_continuity'])"
    )

    results = set()
    for hash_seed in ("0", "1", "2"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert completed.returncode == 0, completed.stderr
        results.add(completed.stdout.strip())

    assert len(results) == 1, f"same seed gave different results across processes: {results}"
