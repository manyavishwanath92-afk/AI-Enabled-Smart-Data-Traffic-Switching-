"""Population bootstrap and per-tick demand generation (Guide Steps 4, 8).

`generate_requests` supports two traffic patterns:

- "uniform": UEs pick a segment weighted by the content server's current
  popularity, independent of cell — demand stays dispersed across cells.
- "flash_crowd": a large fraction of UEs in ONE cell concentrate on ONE
  segment, creating a genuine broadcast opportunity, per the guide's
  explicit requirement that "at least some scenarios must create a large
  same-content group so broadcast has a genuine opportunity."
"""

from __future__ import annotations

import math
import random

from config.config import SimConfig
from traffic_sim.content.content_server import ContentServer
from traffic_sim.content.subscription_manager import SubscriptionManager
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import MobilityState, UE


def create_cells(config: SimConfig) -> list[Cell]:
    """Place `num_cells` cells on a synthetic 2D plane, spaced apart."""
    cells: list[Cell] = []
    for i in range(config.num_cells):
        angle = (2 * math.pi * i) / max(config.num_cells, 1)
        radius = config.cell_area_radius_km * 2.5  # separate cell centers
        cx = radius * math.cos(angle)
        cy = radius * math.sin(angle)
        cells.append(
            Cell(
                cell_id=f"cell_{i}",
                center_x_km=cx,
                center_y_km=cy,
                maximum_capacity_mbps=config.cell_capacity_mbps,
                broadcast_capacity_mbps=config.broadcast_cell_capacity_mbps,
            )
        )
    return cells


def nearest_cell(x_km: float, y_km: float, cells: list[Cell]) -> Cell:
    return min(cells, key=lambda c: (c.center_x_km - x_km) ** 2 + (c.center_y_km - y_km) ** 2)


def create_ues(config: SimConfig, cells: list[Cell], rng: random.Random | None = None) -> list[UE]:
    """Create `num_ues` UEs scattered within cell service areas."""
    rng = rng or random.Random(config.random_seed)
    ues: list[UE] = []
    for i in range(config.num_ues):
        cell = cells[i % len(cells)]
        r = config.cell_area_radius_km * math.sqrt(rng.random())
        theta = rng.uniform(0, 2 * math.pi)
        x = cell.center_x_km + r * math.cos(theta)
        y = cell.center_y_km + r * math.sin(theta)
        speed = rng.uniform(config.ue_mobility_speed_mps_min, config.ue_mobility_speed_mps_max)
        mobility_state = MobilityState.STATIONARY if speed < 0.5 else (
            MobilityState.WALKING if speed < 2.0 else MobilityState.VEHICULAR
        )
        ue = UE(
            ue_id=f"ue_{i}",
            x_km=x,
            y_km=y,
            cell_id=cell.cell_id,
            buffer_seconds=config.initial_buffer_seconds,
            velocity_mps=speed,
            direction_rad=rng.uniform(0, 2 * math.pi),
            mobility_state=mobility_state,
        )
        ues.append(ue)
        cell.connect_ue(ue.ue_id)
    return ues


def generate_requests(
    ues: list[UE],
    content_server: ContentServer,
    subscription_manager: SubscriptionManager,
    scenario: str,
    rng: random.Random,
) -> None:
    """Assign each UE a requested_segment for this tick and record it.

    scenario: "uniform" (popularity-weighted, dispersed) or "flash_crowd"
    (most UEs in one randomly chosen cell request one randomly chosen
    segment; the remainder behave as in "uniform").
    """
    segments = content_server.all_segments()
    segment_ids = [s.segment_id for s in segments]
    weights = [max(s.popularity, 1e-6) for s in segments]

    flash_cell_id: str | None = None
    flash_segment_id: str | None = None
    if scenario == "flash_crowd" and ues:
        flash_cell_id = rng.choice(list({ue.cell_id for ue in ues if ue.cell_id}))
        flash_segment_id = rng.choice(segment_ids)

    for ue in ues:
        if scenario == "flash_crowd" and ue.cell_id == flash_cell_id and rng.random() < 0.85:
            segment_id = flash_segment_id
        else:
            segment_id = rng.choices(segment_ids, weights=weights, k=1)[0]
        ue.requested_segment = segment_id
        subscription_manager.record_request(segment_id, ue.ue_id)
