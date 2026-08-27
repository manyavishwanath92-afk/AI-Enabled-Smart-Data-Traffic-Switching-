"""Runs one simulation in a background thread and keeps it readable live.

The simulator itself is unchanged and unaware of this module. Two existing
public seams are all this needs:

  * `traffic_sim.main.build_simulator()` builds and wires the simulator (the
    AI agent's content-catalogue attachment included), so the API never
    re-implements that wiring;
  * `Simulator.step()` advances exactly one tick and returns that tick's
    `KPIRecord`, so the API can drive the loop itself and observe the run in
    flight — something `Simulator.run()` (which loops to completion) cannot
    offer.

**Capturing the transient per-tick data.** `Simulator.step()` ends with

    stats = mode_selector.current_tick_stats()      # AIDecisions, still alive
    record = kpi_engine.record_tick(..., delivery_results, ..., stats)
    mode_selector.observe(...)                      # clears the decisions

so `record_tick` is the one call that sees both the tick's `DeliveryResult`s
and its `AIDecision`s before either is dropped. This module wraps that method
**on the single KPIEngine instance it just constructed**, captures both, and
delegates to the real implementation. `KPIEngine`, `AIAgent` and `Simulator`
are not modified — the wrapper only reads what is already flowing through an
existing call, and only the latest tick is retained, so memory stays flat.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from config.config import SimConfig
from traffic_sim.api.schemas import (
    decision_record_to_dict,
    decision_to_dict,
    delivery_result_to_dict,
)
from traffic_sim.main import build_simulator
from traffic_sim.simulation.mode_selectors import SELECTOR_FACTORY

SCENARIOS = ("uniform", "flash_crowd")

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"

# Guard rails for a PoC server driven by an untrusted frontend.
MAX_TICKS = 5000
MAX_TICK_INTERVAL_S = 5.0
MAX_CONCURRENT_SIMULATIONS = 20


class SimulationRunner:
    """One simulation, stepped on a background thread, safe to read live."""

    def __init__(
        self,
        sim_id: str,
        config: SimConfig,
        scenario: str,
        selector_name: str,
        total_ticks: int,
        tick_interval_s: float = 0.0,
    ):
        self.sim_id = sim_id
        self.config = config
        self.scenario = scenario
        self.selector_name = selector_name
        self.total_ticks = total_ticks
        self.tick_interval_s = tick_interval_s

        self.simulator = build_simulator(config, selector_name, scenario)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = STATUS_PENDING
        self._error: str | None = None
        self._current_tick = 0

        # Latest tick only — see the module docstring. Long history is already
        # served by KPIEngine.to_records() and AIAgent.decision_history().
        self._latest_decisions: list[dict[str, Any]] = []
        self._latest_groups: list[dict[str, Any]] = []

        # Precomputed inside the sim thread so the (async) WebSocket handler
        # can publish a tick without taking this runner's lock.
        self._latest_payload: dict[str, Any] | None = None

        self._install_tick_capture()

    # --- capture ---------------------------------------------------------

    def _install_tick_capture(self) -> None:
        """Intercept this simulator's own `record_tick` to keep a tick's
        decisions and delivery results before `observe()` discards them."""
        engine = self.simulator.kpi_engine
        original_record_tick = engine.record_tick

        def capturing_record_tick(*args, **kwargs):
            decision_stats = kwargs.get("decision_stats")
            # `delivery_results` is the 4th positional parameter of
            # KPIEngine.record_tick(tick, sim_time_s, ues, delivery_results, ...).
            delivery_results = kwargs.get("delivery_results")
            if delivery_results is None and len(args) >= 4:
                delivery_results = args[3]

            decisions = getattr(decision_stats, "decisions", None) or []
            self._latest_decisions = [decision_to_dict(d) for d in decisions]
            self._latest_groups = [delivery_result_to_dict(r) for r in (delivery_results or [])]

            return original_record_tick(*args, **kwargs)

        engine.record_tick = capturing_record_tick  # type: ignore[method-assign]

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._status = STATUS_RUNNING
            self._thread = threading.Thread(target=self._run, name=f"sim-{self.sim_id}", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    if self._current_tick >= self.total_ticks:
                        break
                    record = self.simulator.step()
                    self._current_tick += 1
                    self._publish_tick(record)
                if self.tick_interval_s > 0:
                    # Interruptible sleep, so stop() takes effect immediately.
                    self._stop_event.wait(self.tick_interval_s)
        except Exception as exc:  # a failed run must be reportable, not silent
            with self._lock:
                self._status = STATUS_FAILED
                self._error = f"{type(exc).__name__}: {exc}"
            return

        with self._lock:
            if self._status == STATUS_RUNNING:
                self._status = STATUS_STOPPED if self._stop_event.is_set() else STATUS_COMPLETED

    def _publish_tick(self, record) -> None:
        """Build the immutable snapshot the WebSocket streams. Lock held.

        Rebound as a whole, never mutated in place, so a reader that grabs it
        without the lock always sees one complete, self-consistent tick.
        """
        from dataclasses import asdict

        self._latest_payload = {
            "type": "tick",
            "simulation_id": self.sim_id,
            "tick": record.tick,
            "kpi": asdict(record),
            "decisions": list(self._latest_decisions),
            "groups": list(self._latest_groups),
        }

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # --- reads (all take the lock; called from FastAPI's threadpool) ------

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def latest_payload(self) -> dict[str, Any] | None:
        """The most recent tick snapshot, or None before the first tick.

        Read without the lock by design — see `_publish_tick`. Its `tick`
        field is what the WebSocket uses to detect a new tick.
        """
        return self._latest_payload

    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "simulation_id": self.sim_id,
                "status": self._status,
                "current_tick": self._current_tick,
                "total_ticks": self.total_ticks,
                "scenario": self.scenario,
                "mode_selector": self.selector_name,
                "tick_interval_s": self.tick_interval_s,
                "seed": self.config.random_seed,
                "error": self._error,
            }

    def kpi_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return self.simulator.kpi_engine.to_records()

    def kpi_summary(self) -> dict[str, Any]:
        with self._lock:
            return self.simulator.kpi_engine.summary()

    def cells(self) -> list:
        with self._lock:
            return list(self.simulator.cells)

    def ues(self) -> list:
        with self._lock:
            return list(self.simulator.ues)

    def latest_decisions(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._latest_decisions)

    def latest_groups(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._latest_groups)

    def decision_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """The AI agent's persisted switching history.

        Empty for fixed-rule selectors: they have no agent, and deliberately
        never pass through MBOS/PCF or record decisions.
        """
        history_fn = getattr(self.simulator.mode_selector, "decision_history", None)
        if not callable(history_fn):
            return []
        with self._lock:
            return [decision_record_to_dict(r) for r in history_fn(limit)]


class SimulationRegistry:
    """The set of simulations this server is currently holding."""

    def __init__(self, max_simulations: int = MAX_CONCURRENT_SIMULATIONS):
        self._runners: dict[str, SimulationRunner] = {}
        self._lock = threading.RLock()
        self._max = max_simulations

    def create(
        self,
        config: SimConfig,
        scenario: str,
        selector_name: str,
        total_ticks: int,
        tick_interval_s: float = 0.0,
    ) -> SimulationRunner:
        sim_id = f"sim_{uuid.uuid4().hex[:10]}"
        runner = SimulationRunner(
            sim_id=sim_id,
            config=config,
            scenario=scenario,
            selector_name=selector_name,
            total_ticks=total_ticks,
            tick_interval_s=tick_interval_s,
        )
        with self._lock:
            self._evict_if_needed()
            self._runners[sim_id] = runner
        return runner

    def _evict_if_needed(self) -> None:
        """Drop the oldest finished run when at capacity. Lock held."""
        while len(self._runners) >= self._max:
            finished = [
                sid
                for sid, r in self._runners.items()
                if r.status in (STATUS_COMPLETED, STATUS_STOPPED, STATUS_FAILED)
            ]
            victim = finished[0] if finished else next(iter(self._runners))
            self._runners.pop(victim).stop()

    def get(self, sim_id: str) -> SimulationRunner | None:
        with self._lock:
            return self._runners.get(sim_id)

    def list(self) -> list[SimulationRunner]:
        with self._lock:
            return list(self._runners.values())

    def remove(self, sim_id: str) -> bool:
        with self._lock:
            runner = self._runners.pop(sim_id, None)
        if runner is None:
            return False
        runner.stop()
        return True

    def shutdown(self) -> None:
        for runner in self.list():
            runner.stop()


def validate_selector(name: str) -> None:
    if name not in SELECTOR_FACTORY:
        raise ValueError(f"unknown mode_selector {name!r}; expected one of {sorted(SELECTOR_FACTORY)}")


def validate_scenario(name: str) -> None:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; expected one of {list(SCENARIOS)}")


def wait_for_status(runner: SimulationRunner, statuses: tuple[str, ...], timeout: float = 30.0) -> str:
    """Block until the run reaches one of `statuses` (or the timeout lapses).

    Used by tests and by any caller that wants a completed run rather than a
    live one; the HTTP layer itself never blocks on this.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = runner.status
        if status in statuses:
            return status
        time.sleep(0.01)
    return runner.status
