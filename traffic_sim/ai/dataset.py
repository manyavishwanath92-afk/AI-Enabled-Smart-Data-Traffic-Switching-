"""Manufacture training data by running the simulator.

For every (cell, segment) group the simulator encounters, we record the
decision-time features and BOTH counterfactual outcomes — what unicast
would have delivered and what broadcast would have delivered.

Getting both answers is only possible because this is a simulation, and
it only happens offline, at training time. The trained model's job is to
reproduce those answers at decision time from features alone.

State diversity comes from sweeping seeds, both traffic scenarios, and a
range of CAPACITY SETTINGS, with runs driven by a randomising selector.

Sweeping capacity matters more than it looks. A gradient-boosted tree
cannot extrapolate beyond the values it was trained on: if every training
row carried the same broadcast capacity, the model would learn to ignore
that feature entirely and would keep predicting the trained capacity's
outcome even when the real carrier was a fraction of the size. The
decision layer would then confidently choose broadcast into a starved
carrier. Varying capacity is what makes the feature informative and the
model honest outside the default configuration.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from config.config import SimConfig
from traffic_sim.ai.feature_engineering import FEATURE_NAMES, extract_features
from traffic_sim.ai.outcomes import evaluate_modes
from traffic_sim.simulation.mode_selectors import make_random_selector
from traffic_sim.simulation.simulator import Simulator

LABEL_NAMES = ["y_unicast_mbps", "y_broadcast_mbps"]


@dataclass
class Dataset:
    rows: list[dict[str, float]]

    def __len__(self) -> int:
        return len(self.rows)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = FEATURE_NAMES + LABEL_NAMES
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)

    @classmethod
    def from_csv(cls, path: str | Path) -> "Dataset":
        with Path(path).open() as handle:
            return cls(rows=[{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)])

    def xy(self) -> tuple[list[list[float]], list[float], list[float]]:
        x = [[row[name] for name in FEATURE_NAMES] for row in self.rows]
        y_unicast = [row["y_unicast_mbps"] for row in self.rows]
        y_broadcast = [row["y_broadcast_mbps"] for row in self.rows]
        return x, y_unicast, y_broadcast


# Capacity settings the sweep visits, so the models learn how capacity
# actually affects achievable rate instead of treating it as a constant.
BROADCAST_CAPACITIES_MBPS: tuple[float, ...] = (2.0, 8.0, 20.0, 60.0)
CELL_CAPACITIES_MBPS: tuple[float, ...] = (40.0, 100.0)


def generate_dataset(
    base_config: SimConfig | None = None,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    scenarios: tuple[str, ...] = ("flash_crowd", "uniform"),
    ticks: int = 40,
    broadcast_capacities: tuple[float, ...] = BROADCAST_CAPACITIES_MBPS,
    cell_capacities: tuple[float, ...] = CELL_CAPACITIES_MBPS,
) -> Dataset:
    """Sweep scenarios, seeds and capacities, one row per group per tick."""
    base_config = base_config or SimConfig.default()
    rows: list[dict[str, float]] = []

    for seed in seeds:
        for scenario in scenarios:
            # Rotate capacity settings across seeds so the sweep covers the
            # grid without multiplying run count by the full cross-product.
            broadcast_capacity = broadcast_capacities[seed % len(broadcast_capacities)]
            cell_capacity = cell_capacities[seed % len(cell_capacities)]
            config = base_config.with_overrides(
                random_seed=seed,
                broadcast_cell_capacity_mbps=broadcast_capacity,
                cell_capacity_mbps=cell_capacity,
            )
            rng = random.Random(seed)
            simulator = Simulator(config, make_random_selector(rng), scenario=scenario)

            for _ in range(ticks):
                simulator.step()
                # Sample the post-tick state: buffers and congestion have
                # moved, giving a fresh situation to label.
                rows.extend(_rows_for_current_state(simulator, config))

    return Dataset(rows=rows)


def _rows_for_current_state(simulator: Simulator, config: SimConfig) -> list[dict[str, float]]:
    cells_by_id = {c.cell_id: c for c in simulator.cells}
    groups: dict[tuple[str, str], list] = {}
    for ue in simulator.ues:
        if ue.cell_id is None or ue.requested_segment is None:
            continue
        groups.setdefault((ue.cell_id, ue.requested_segment), []).append(ue)

    # How many groups each cell currently has on its broadcast carrier —
    # the same contention signal the agent sees at decision time.
    broadcast_groups: dict[str, int] = {}
    for ue in simulator.ues:
        if ue.cell_id and ue.delivery_mode.value == "broadcast" and ue.requested_segment:
            broadcast_groups.setdefault(ue.cell_id, set()).add(ue.requested_segment)  # type: ignore[arg-type]
    broadcast_counts = {cell_id: len(segments) for cell_id, segments in broadcast_groups.items()}

    rows = []
    for (cell_id, segment_id), ues in groups.items():
        cell = cells_by_id[cell_id]
        segment = simulator.content_server.get_segment(segment_id)
        features = extract_features(
            segment,
            ues,
            cell,
            config,
            prev_congestion=cell.congestion(),
            prev_broadcast_groups=broadcast_counts.get(cell_id, 0),
        )
        outcomes = evaluate_modes(features, config)

        row = dict(features.values)
        row["y_unicast_mbps"] = outcomes.unicast.achievable_mbps
        row["y_broadcast_mbps"] = outcomes.broadcast.achievable_mbps
        rows.append(row)
    return rows
