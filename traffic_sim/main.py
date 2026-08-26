"""CLI for running the simulator and comparing delivery policies.

Examples:
    python -m traffic_sim.main --scenario flash_crowd --mode-selector ai
    python -m traffic_sim.main --compare
"""

from __future__ import annotations

import argparse
import random

from config.config import SimConfig
from traffic_sim.experiments.compare import run_comparison
from traffic_sim.simulation.mode_selectors import SELECTOR_FACTORY
from traffic_sim.simulation.simulator import Simulator


def build_simulator(config: SimConfig, selector_name: str, scenario: str) -> Simulator:
    """Build a simulator, wiring the AI agent to the content catalogue if needed."""
    rng = random.Random(config.random_seed)
    selector = SELECTOR_FACTORY[selector_name](rng, config)
    simulator = Simulator(config, selector, scenario=scenario)

    # The AI agent needs the segment catalogue, which the Simulator owns.
    attach = getattr(selector, "attach", None)
    if callable(attach):
        attach(simulator.content_server)

    return simulator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AI-enabled traffic-switching simulator.")
    parser.add_argument("--ticks", type=int, default=None, help="Number of ticks to run (overrides --duration).")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: config).")
    parser.add_argument("--scenario", choices=["uniform", "flash_crowd"], default="flash_crowd")
    parser.add_argument(
        "--mode-selector",
        choices=list(SELECTOR_FACTORY.keys()),
        default="ai",
        help="Delivery-mode policy. 'ai' is the XGBoost + utility + agent stack.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="data/generated/run.csv")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run every policy on matched seeds and print a KPI comparison.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    overrides: dict = {}
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    config = SimConfig.default().with_overrides(**overrides) if overrides else SimConfig.default()

    if args.compare:
        ticks = args.ticks or 60
        run_comparison(config, scenario=args.scenario, ticks=ticks, print_table=True)
        return

    simulator = build_simulator(config, args.mode_selector, args.scenario)

    duration_s = args.duration
    if args.ticks is not None:
        duration_s = args.ticks * config.simulation_timestep_s
    simulator.run(duration_s)

    simulator.kpi_engine.export_csv(args.out)

    summary = simulator.kpi_engine.summary()
    print(f"Ran {summary.get('num_ticks', 0)} ticks | scenario={args.scenario} | mode_selector={args.mode_selector}")

    selector = simulator.mode_selector
    if hasattr(selector, "uses_model"):
        source = "XGBoost model" if selector.uses_model else "analytical fallback (no model found)"
        print(f"  AI prediction source: {source}")

    for key, value in summary.items():
        if key == "num_ticks":
            continue
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"KPI records exported to {args.out}")


if __name__ == "__main__":
    main()
