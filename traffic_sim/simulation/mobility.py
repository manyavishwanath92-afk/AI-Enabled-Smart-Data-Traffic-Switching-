"""UE movement, cell reassociation, and mobility risk (Guide Steps 5, 10.4, 18).

Mobility prediction is deliberately a transparent, deterministic score —
NOT a deep-learning predictor. MobilityRisk is proportional to
1 / expected_time_in_coverage, where expected_time_in_coverage is
estimated from distance-to-cell-edge and speed. Reinforcement learning
and deep-learning mobility prediction are out of scope entirely (see
traffic_sim/ai/__init__.py).
"""

from __future__ import annotations

import math

from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.traffic import nearest_cell
from traffic_sim.simulation.user import MobilityState, UE

# Stationary/very-slow UEs are treated as effectively infinite coverage
# time (near-zero risk) rather than raising a division-by-zero.
MIN_VELOCITY_MPS = 1e-6


def update_positions(ues: list[UE], dt: float) -> None:
    for ue in ues:
        if ue.mobility_state == MobilityState.STATIONARY or ue.velocity_mps <= 0:
            continue
        ue.x_km += (ue.velocity_mps * math.cos(ue.direction_rad) * dt) / 1000.0
        ue.y_km += (ue.velocity_mps * math.sin(ue.direction_rad) * dt) / 1000.0


def reassociate_cells(ues: list[UE], cells: list[Cell]) -> None:
    for ue in ues:
        new_cell = nearest_cell(ue.x_km, ue.y_km, cells)
        if new_cell.cell_id != ue.cell_id:
            for c in cells:
                if c.cell_id == ue.cell_id:
                    c.disconnect_ue(ue.ue_id)
            new_cell.connect_ue(ue.ue_id)
            ue.cell_id = new_cell.cell_id


def compute_mobility_score(ue: UE, cell: Cell, cell_area_radius_km: float) -> float:
    """MobilityRisk ∝ 1 / expected_time_in_coverage.

    expected_time_in_coverage ≈ (distance_to_cell_edge_km * 1000) / velocity_mps,
    capped so stationary/near-stationary UEs get risk ≈ 0 rather than inf.
    """
    distance_from_center_km = math.hypot(ue.x_km - cell.center_x_km, ue.y_km - cell.center_y_km)
    distance_to_edge_km = max(cell_area_radius_km - distance_from_center_km, 0.0)
    velocity = max(ue.velocity_mps, MIN_VELOCITY_MPS)
    expected_time_in_coverage_s = (distance_to_edge_km * 1000.0) / velocity
    if expected_time_in_coverage_s <= 0:
        risk = 1.0  # already at/beyond the edge
    else:
        risk = 1.0 / expected_time_in_coverage_s
    ue.mobility_risk = min(risk, 1.0) if ue.mobility_state != MobilityState.STATIONARY else 0.0
    return ue.mobility_risk
