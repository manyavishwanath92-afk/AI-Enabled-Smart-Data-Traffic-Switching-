"""FastAPI app serving the dashboard contract in docs/FRONTEND_API_HANDOFF.md.

Every value returned here is produced by the simulator, the KPI engine, the AI
agent or the comparison harness — this layer only routes and serializes. No
metric is computed, defaulted or synthesized in this file.

Threading note: the REST handlers are plain `def`, so FastAPI runs them in its
worker threadpool, where blocking briefly on a runner's lock is safe. Only the
WebSocket handler is `async`, and it touches nothing that locks — it reads the
immutable per-tick snapshot the simulation thread publishes.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.config import SimConfig
from traffic_sim.api.runner import (
    MAX_TICK_INTERVAL_S,
    MAX_TICKS,
    SCENARIOS,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_STOPPED,
    SimulationRegistry,
    SimulationRunner,
    validate_scenario,
    validate_selector,
)
from traffic_sim.api.schemas import cell_to_dict, ue_to_dict
from traffic_sim.experiments.compare import run_comparison
from traffic_sim.simulation.mode_selectors import SELECTOR_FACTORY

# How often the WebSocket checks for a new tick. Well below a typical
# tick_interval_s, so the stream stays smooth without busy-spinning.
_WS_POLL_INTERVAL_S = 0.02

TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_STOPPED, STATUS_FAILED)


class CreateSimulationRequest(BaseModel):
    scenario: str = "flash_crowd"
    mode_selector: str = "ai"
    ticks: int = Field(default=120, ge=1, le=MAX_TICKS)
    seed: int | None = None
    # Seconds to pause between ticks. 0 runs flat out (best for "give me a
    # finished run"); ~0.1-0.5 makes a run watchable live on the dashboard.
    tick_interval_s: float = Field(default=0.0, ge=0.0, le=MAX_TICK_INTERVAL_S)
    # Any SimConfig field, e.g. {"broadcast_cell_capacity_mbps": 4.0} to demo
    # a starved carrier. Validated by SimConfig itself.
    config_overrides: dict[str, Any] | None = None


def _build_config(request: CreateSimulationRequest) -> SimConfig:
    overrides: dict[str, Any] = dict(request.config_overrides or {})
    if request.seed is not None:
        overrides["random_seed"] = request.seed

    base = SimConfig.default()
    if not overrides:
        return base

    unknown = set(overrides) - set(base.to_dict())
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    # with_overrides() validates; a bad value raises ValueError, surfaced as 400.
    return base.with_overrides(**overrides)


def create_app() -> FastAPI:
    registry = SimulationRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        # Simulation threads are daemons, but stopping them explicitly means a
        # shutdown doesn't leave a run stepping into a half-torn-down process.
        registry.shutdown()

    app = FastAPI(
        title="AI-Enabled Smart Data Traffic Switching API",
        description=(
            "Live simulator data for the dashboard. Contract and field-by-field "
            "sourcing: docs/FRONTEND_API_HANDOFF.md"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # The dashboard is served from a different origin during development, so
    # the browser needs CORS permission to call this at all. Wide open is
    # acceptable here because this is a local PoC serving simulated data and
    # holding no credentials; a deployed instance should name its origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.registry = registry

    def _runner_or_404(simulation_id: str) -> SimulationRunner:
        runner = registry.get(simulation_id)
        if runner is None:
            raise HTTPException(status_code=404, detail=f"unknown simulation_id {simulation_id!r}")
        return runner

    # --- meta ------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "simulations": len(registry.list())}

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        """The default SimConfig — every knob the simulator runs on."""
        return SimConfig.default().to_dict()

    @app.get("/api/scenarios")
    def get_scenarios() -> dict[str, Any]:
        """Valid `scenario` / `mode_selector` values, so the frontend's
        controls come from the backend rather than a hardcoded list."""
        return {
            "scenarios": list(SCENARIOS),
            "mode_selectors": sorted(SELECTOR_FACTORY),
            "ai_mode_selector": "ai",
            "max_ticks": MAX_TICKS,
            "max_tick_interval_s": MAX_TICK_INTERVAL_S,
        }

    # --- simulation lifecycle --------------------------------------------

    @app.post("/api/simulations", status_code=201)
    def create_simulation(request: CreateSimulationRequest) -> dict[str, Any]:
        try:
            validate_scenario(request.scenario)
            validate_selector(request.mode_selector)
            config = _build_config(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        runner = registry.create(
            config=config,
            scenario=request.scenario,
            selector_name=request.mode_selector,
            total_ticks=request.ticks,
            tick_interval_s=request.tick_interval_s,
        )
        runner.start()
        return runner.status_dict()

    @app.get("/api/simulations")
    def list_simulations() -> list[dict[str, Any]]:
        return [runner.status_dict() for runner in registry.list()]

    @app.get("/api/simulations/{simulation_id}/status")
    def simulation_status(simulation_id: str) -> dict[str, Any]:
        return _runner_or_404(simulation_id).status_dict()

    @app.post("/api/simulations/{simulation_id}/pause")
    def pause_simulation(simulation_id: str) -> dict[str, Any]:
        """Hold a running simulation between ticks.

        An open WebSocket stays connected across a pause — `paused` is not a
        terminal status — so the dashboard keeps its live view and simply
        stops receiving new ticks until resumed.
        """
        runner = _runner_or_404(simulation_id)
        if not runner.pause():
            raise HTTPException(
                status_code=409, detail=f"simulation is {runner.status}, not running"
            )
        return runner.status_dict()

    @app.post("/api/simulations/{simulation_id}/resume")
    def resume_simulation(simulation_id: str) -> dict[str, Any]:
        runner = _runner_or_404(simulation_id)
        if not runner.resume():
            raise HTTPException(
                status_code=409, detail=f"simulation is {runner.status}, not paused"
            )
        return runner.status_dict()

    @app.post("/api/simulations/{simulation_id}/stop")
    def stop_simulation(simulation_id: str) -> dict[str, Any]:
        runner = _runner_or_404(simulation_id)
        runner.stop()
        runner.join(timeout=5.0)
        return runner.status_dict()

    @app.delete("/api/simulations/{simulation_id}")
    def delete_simulation(simulation_id: str) -> dict[str, Any]:
        if not registry.remove(simulation_id):
            raise HTTPException(status_code=404, detail=f"unknown simulation_id {simulation_id!r}")
        return {"simulation_id": simulation_id, "deleted": True}

    # --- KPIs ------------------------------------------------------------

    @app.get("/api/simulations/{simulation_id}/kpis/latest")
    def kpis_latest(simulation_id: str) -> dict[str, Any]:
        records = _runner_or_404(simulation_id).kpi_records()
        if not records:
            raise HTTPException(status_code=404, detail="no ticks recorded yet")
        return records[-1]

    @app.get("/api/simulations/{simulation_id}/kpis/history")
    def kpis_history(
        simulation_id: str,
        from_tick: int | None = Query(default=None, ge=0),
        to_tick: int | None = Query(default=None, ge=0),
    ) -> list[dict[str, Any]]:
        records = _runner_or_404(simulation_id).kpi_records()
        if from_tick is not None:
            records = [r for r in records if r["tick"] >= from_tick]
        if to_tick is not None:
            records = [r for r in records if r["tick"] <= to_tick]
        return records

    @app.get("/api/simulations/{simulation_id}/kpis/summary")
    def kpis_summary(simulation_id: str) -> dict[str, Any]:
        return _runner_or_404(simulation_id).kpi_summary()

    # --- network state ---------------------------------------------------

    @app.get("/api/simulations/{simulation_id}/cells")
    def cells(simulation_id: str) -> list[dict[str, Any]]:
        return [cell_to_dict(c) for c in _runner_or_404(simulation_id).cells()]

    @app.get("/api/simulations/{simulation_id}/ues")
    def ues(
        simulation_id: str,
        cell_id: str | None = None,
        limit: int = Query(default=200, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        items = _runner_or_404(simulation_id).ues()
        if cell_id is not None:
            items = [u for u in items if u.cell_id == cell_id]
        return {
            "total": len(items),
            "limit": limit,
            "offset": offset,
            "items": [ue_to_dict(u) for u in items[offset : offset + limit]],
        }

    @app.get("/api/simulations/{simulation_id}/groups/latest")
    def groups_latest(simulation_id: str) -> list[dict[str, Any]]:
        """Per-group delivery detail for the most recent tick."""
        return _runner_or_404(simulation_id).latest_groups()

    # --- AI decisions ----------------------------------------------------

    @app.get("/api/simulations/{simulation_id}/decisions/latest")
    def decisions_latest(simulation_id: str) -> list[dict[str, Any]]:
        """Full-detail decisions for the most recent tick, `reason` included.

        Empty for fixed-rule selectors — they make no explainable decision.
        """
        return _runner_or_404(simulation_id).latest_decisions()

    @app.get("/api/simulations/{simulation_id}/decisions/history")
    def decisions_history(
        simulation_id: str,
        limit: int | None = Query(default=None, ge=1),
    ) -> list[dict[str, Any]]:
        """Persisted per-group switching history (AI selector only)."""
        return _runner_or_404(simulation_id).decision_history(limit)

    # --- comparison ------------------------------------------------------

    @app.get("/api/comparison")
    def comparison(
        scenario: str = "flash_crowd",
        ticks: int = Query(default=60, ge=1, le=500),
        seeds: list[int] = Query(default=[11, 22, 33]),
    ) -> list[dict[str, Any]]:
        """Every policy on matched seeds — the AI-vs-baseline table.

        Synchronous and CPU-bound: it runs 4 policies x len(seeds) full
        simulations before returning, so keep `ticks` and `seeds` modest.
        """
        try:
            validate_scenario(scenario)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not seeds:
            raise HTTPException(status_code=400, detail="at least one seed is required")

        rows = run_comparison(scenario=scenario, ticks=ticks, seeds=tuple(seeds))
        return [{"policy": row.policy, "summary": row.summary} for row in rows]

    # --- live updates ----------------------------------------------------

    @app.websocket("/api/simulations/{simulation_id}/live")
    async def live(websocket: WebSocket, simulation_id: str) -> None:
        """One message per simulated tick, then a final status message.

        Reads only the immutable snapshot the simulation thread publishes, so
        the event loop is never blocked by a tick in progress.
        """
        await websocket.accept()
        runner = registry.get(simulation_id)
        if runner is None:
            await websocket.send_json({"type": "error", "detail": f"unknown simulation_id {simulation_id!r}"})
            await websocket.close()
            return

        # Tracked by the payload's own tick number rather than a separate
        # counter: the snapshot dict is rebound atomically by the simulation
        # thread, so reading it once cannot tear, and tick numbers only ever
        # increase — no message can be duplicated or skipped.
        last_sent_tick = 0
        try:
            while True:
                payload = runner.latest_payload
                if payload is not None and payload["tick"] != last_sent_tick:
                    last_sent_tick = payload["tick"]
                    await websocket.send_json(payload)
                    continue  # a tick may already be waiting; drain first
                if runner.status in TERMINAL_STATUSES:
                    break
                await asyncio.sleep(_WS_POLL_INTERVAL_S)

            await websocket.send_json({"type": "status", **runner.status_dict()})
        except WebSocketDisconnect:
            return
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass  # already closed by the client

    return app


app = create_app()
