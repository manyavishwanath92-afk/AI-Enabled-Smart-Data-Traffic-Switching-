"""Run every delivery policy on matched scenarios and compare KPIs.

The guide's mandatory experiment (Section 22): the same traffic, the same
seeds, the same content — only the delivery policy differs. That is what
makes any difference in the results attributable to the policy rather
than to luck.

Success is not "broadcast more". It is measurably fewer resources spent
while service continuity and QoE hold up.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.config import SimConfig

POLICIES = ("always-unicast", "always-broadcast", "rule-by-group-size", "ai")

REPORTED_KPIS = (
    ("resource_savings_vs_baseline", "resource saved"),
    ("service_continuity", "service continuity"),
    ("mean_qoe", "QoE"),
    ("mean_congestion", "congestion"),
    ("buffer_starvation_rate", "starvation"),
    ("mean_achievable_mbps", "achieved Mbps"),
    ("total_switches", "switches"),
    ("mean_inference_latency_ms", "inference ms"),
)


@dataclass
class ComparisonRow:
    policy: str
    summary: dict


def run_comparison(
    config: SimConfig | None = None,
    scenario: str = "flash_crowd",
    ticks: int = 60,
    seeds: tuple[int, ...] = (11, 22, 33),
    print_table: bool = False,
) -> list[ComparisonRow]:
    """Average each policy's KPIs across matched seeds."""
    # Imported here to avoid a circular import at module load time.
    from traffic_sim.main import build_simulator

    base = config or SimConfig.default()
    rows: list[ComparisonRow] = []

    for policy in POLICIES:
        totals: dict[str, float] = {}
        for seed in seeds:
            seeded = base.with_overrides(random_seed=seed)
            simulator = build_simulator(seeded, policy, scenario)
            simulator.run(duration_s=ticks * seeded.simulation_timestep_s)
            summary = simulator.kpi_engine.summary()
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0.0) + float(value)

        averaged = {key: value / len(seeds) for key, value in totals.items()}
        rows.append(ComparisonRow(policy=policy, summary=averaged))

    if print_table:
        _print_table(rows, scenario, ticks, seeds)

    return rows


def _print_table(rows: list[ComparisonRow], scenario: str, ticks: int, seeds: tuple[int, ...]) -> None:
    print(f"\nMatched comparison | scenario={scenario} | {ticks} ticks | seeds={list(seeds)}\n")
    header = f"{'policy':<20}" + "".join(f"{label:>20}" for _, label in REPORTED_KPIS)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row.policy:<20}"
        for key, _ in REPORTED_KPIS:
            value = row.summary.get(key, 0.0)
            line += f"{value:>20.4f}"
        print(line)
    print()
