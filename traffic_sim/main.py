"""CLI entry point for manually running the Phase 1 simulator (Guide Step 20).

Example:
    python -m traffic_sim.main --scenario flash_crowd --mode-selector always-broadcast
    python -m traffic_sim.main --scenario flash_crowd --mode-selector always-unicast
"""

from __future__ import annotations

import argparse
import random

from config.config import SimConfig
from traffic_sim.simulation.mode_selectors import SELECTOR_FACTORY
from traffic_sim.simulation.simulator import Simulator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 1 network simulator (no AI).")
    parser.add_argument("--ticks", type=int, default=None, help="Number of ticks to run (overrides --duration).")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: config).")
    parser.add_argument("--scenario", choices=["uniform", "flash_crowd"], default="flash_crowd")
    parser.add_argument(
        "--mode-selector",
        choices=list(SELECTOR_FACTORY.keys()),
        default="rule-by-group-size",
        help="Fixed, non-AI delivery-mode rule (Phase 1 only; the AI agent replaces this in Phase 2).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="data/generated/run.csv")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    overrides: dict = {}
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    config = SimConfig.default().with_overrides(**overrides) if overrides else SimConfig.default()

    rng = random.Random(config.random_seed)
    selector = SELECTOR_FACTORY[args.mode_selector](rng)

    simulator = Simulator(config, selector, scenario=args.scenario)

    duration_s = args.duration
    if args.ticks is not None:
        duration_s = args.ticks * config.simulation_timestep_s
    simulator.run(duration_s)

    simulator.kpi_engine.export_csv(args.out)

    summary = simulator.kpi_engine.summary()
    print(f"Ran {summary.get('num_ticks', 0)} ticks | scenario={args.scenario} | mode_selector={args.mode_selector}")
    for key, value in summary.items():
        if key == "num_ticks":
            continue
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"KPI records exported to {args.out}")


if __name__ == "__main__":
    main()
