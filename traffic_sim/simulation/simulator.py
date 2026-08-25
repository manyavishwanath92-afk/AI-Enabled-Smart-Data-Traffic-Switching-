"""Simulator: the per-tick orchestration loop (Guide Section 28).

Delivery-mode selection is injected via a `mode_selector` callable rather
than decided internally. In Phase 1 this is always a fixed, non-AI rule
(always-unicast, always-broadcast, random, or a naive group-size
threshold — see traffic_sim.main). This keeps a clean seam: Phase 2's AI
agent will later be plugged in as just another `mode_selector`
implementation, with no change required to Simulator itself.
"""

from __future__ import annotations

import random
from typing import Callable

from config.config import SimConfig
from traffic_sim.content.content_server import ContentServer
from traffic_sim.content.subscription_manager import SubscriptionManager
from traffic_sim.delivery.broadcast import BroadcastDelivery
from traffic_sim.delivery.result import DeliveryResult
from traffic_sim.delivery.unicast import UnicastDelivery
from traffic_sim.kpi.metrics import KPIEngine, KPIRecord
from traffic_sim.simulation import mobility, traffic
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import DeliveryMode, UE

# (segment_id, requesting UEs in one cell, that cell, config) -> chosen mode.
# Phase 1 implementations are fixed rules only — see traffic_sim.main for
# the concrete selectors used by the CLI/tests.
ModeSelector = Callable[[str, list[UE], Cell, SimConfig], DeliveryMode]


class Simulator:
    def __init__(self, config: SimConfig, mode_selector: ModeSelector, scenario: str = "uniform"):
        config.validate()
        self.config = config
        self.mode_selector = mode_selector
        self.scenario = scenario

        self._rng = random.Random(config.random_seed)
        self.cells: list[Cell] = traffic.create_cells(config)
        self.ues: list[UE] = traffic.create_ues(config, self.cells, self._rng)
        self.content_server = ContentServer(config, self._rng)
        self.subscription_manager = SubscriptionManager(self.content_server)
        self.kpi_engine = KPIEngine(config)

        self._unicast = UnicastDelivery()
        self._broadcast = BroadcastDelivery()
        self._tick = 0
        self._sim_time_s = 0.0

    def step(self) -> KPIRecord:
        cfg = self.config
        dt = cfg.simulation_timestep_s

        mobility.update_positions(self.ues, dt)
        mobility.reassociate_cells(self.ues, self.cells)

        cells_by_id = {c.cell_id: c for c in self.cells}
        for ue in self.ues:
            cell = cells_by_id.get(ue.cell_id)
            if cell is not None:
                mobility.compute_mobility_score(ue, cell, cfg.cell_area_radius_km)

        for cell in self.cells:
            cell.reset_load()

        self.subscription_manager.clear_requests()
        self.content_server.drift_popularity(dt)
        traffic.generate_requests(self.ues, self.content_server, self.subscription_manager, self.scenario, self._rng)
        self.subscription_manager.compute_popularity(len(self.ues))

        # Group requesting UEs by (cell_id, segment_id).
        groups: dict[tuple[str, str], list[UE]] = {}
        for ue in self.ues:
            if ue.cell_id is None or ue.requested_segment is None:
                continue
            groups.setdefault((ue.cell_id, ue.requested_segment), []).append(ue)

        # Pass 1 — decide every group's delivery mode BEFORE delivering any
        # of them, so we can count how many UEs each cell is actually
        # serving by unicast. UEs offloaded to broadcast must not count
        # against the cellular resource share of the UEs that remain on
        # unicast; that offload relief is a core claim of the project and
        # only becomes measurable if modes are known up front.
        decisions: list[tuple[str, str, list[UE], DeliveryMode]] = []
        unicast_users_by_cell: dict[str, int] = {c.cell_id: 0 for c in self.cells}
        for (cell_id, segment_id), group_ues in groups.items():
            cell = cells_by_id[cell_id]
            mode = self.mode_selector(segment_id, group_ues, cell, cfg)
            decisions.append((cell_id, segment_id, group_ues, mode))
            if mode != DeliveryMode.BROADCAST:
                unicast_users_by_cell[cell_id] += len(group_ues)

        # Pass 2 — deliver, with the correct per-cell unicast sharing count.
        delivery_results: list[DeliveryResult] = []
        for cell_id, segment_id, group_ues, mode in decisions:
            cell = cells_by_id[cell_id]
            segment = self.content_server.get_segment(segment_id)
            active_users_in_cell = len(cell.connected_ue_ids)
            unicast_users_in_cell = unicast_users_by_cell[cell_id]

            deliverer = self._broadcast if mode == DeliveryMode.BROADCAST else self._unicast
            result = deliverer.deliver(
                group_ues, segment, cell, cfg, dt, active_users_in_cell, unicast_users_in_cell
            )

            cell.add_load(result.required_resource_mbps)
            delivery_results.append(result)

        self._tick += 1
        self._sim_time_s += dt
        return self.kpi_engine.record_tick(self._tick, self._sim_time_s, self.ues, delivery_results, dt)

    def run(self, duration_s: float | None = None) -> KPIEngine:
        duration_s = duration_s if duration_s is not None else self.config.simulation_duration_s
        n_ticks = int(duration_s / self.config.simulation_timestep_s)
        for _ in range(n_ticks):
            self.step()
        return self.kpi_engine
