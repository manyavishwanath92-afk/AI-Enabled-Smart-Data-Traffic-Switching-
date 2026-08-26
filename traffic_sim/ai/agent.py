"""The AI decision agent — the final say on UNICAST vs BROADCAST.

It is not the model, and it is not the utility function. Those answer
"what will happen" and "which outcome is better". The agent answers a
third question: "what are we actually allowed to do right now?"

Order of reasoning, per (cell, segment) group, per tick:

  1. read telemetry            -> ai.feature_engineering
  2. predict both outcomes     -> ai.predict (XGBoost, or analytical)
  3. score both outcomes       -> ai.utility
  4. pick the better score
  5. refuse to act if unsafe:
       a. buffer safety - too little margin to survive a switch
       b. low confidence - the two options are nearly tied
       c. hysteresis    - the gain is real but too small to be worth churn
       d. cooldown      - this group switched too recently

Steps 5a-5d are why the agent exists. Without them a marginally better
score would flip a group's delivery mode every tick, which costs more in
churn than it gains in throughput.

`AIAgent` is a callable object with the exact `ModeSelectorFn` signature,
so it drops into the simulator's existing seam with no interface change.
It also exposes `observe()`, which the simulator calls after each tick —
that is the closed loop: what happened feeds the next decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.config import SimConfig
from traffic_sim.ai.feature_engineering import extract_features
from traffic_sim.ai.predict import OutcomePredictor
from traffic_sim.ai.utility import UtilityBreakdown, compute_utility, resolve_weights
from traffic_sim.content.content_server import ContentServer
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import DeliveryMode, UE


@dataclass
class AIDecision:
    """One explainable decision."""

    mode: DeliveryMode
    confidence: float
    reason: dict[str, object]
    utility_unicast: float
    utility_broadcast: float
    predicted_unicast_mbps: float
    predicted_broadcast_mbps: float
    prediction_source: str
    inference_latency_ms: float
    switched: bool


@dataclass
class _GroupState:
    """What the agent remembers about one group between ticks."""

    current_mode: DeliveryMode | None = None
    last_switch_time_s: float | None = None


@dataclass
class TickStats:
    """Per-tick decision telemetry handed to the KPI engine."""

    switch_count: int = 0
    mean_inference_latency_ms: float = 0.0
    decisions: list[AIDecision] = field(default_factory=list)


class AIAgent:
    def __init__(
        self,
        config: SimConfig,
        content_server: ContentServer | None = None,
        model_dir: str | None = None,
    ):
        self.config = config
        self._content_server = content_server
        self.predictor = OutcomePredictor(config, model_dir)
        self._groups: dict[tuple[str, str], _GroupState] = {}
        self._prev_congestion: dict[str, float] = {}
        self._prev_broadcast_groups: dict[str, int] = {}
        self._sim_time_s = 0.0
        self._tick_decisions: list[AIDecision] = []

    def attach(self, content_server: ContentServer) -> None:
        """Give the agent the content catalogue.

        The simulator builds its own ContentServer, so the agent is
        constructed first and wired up afterwards. Without it the agent
        still runs, using a mid-range bitrate assumption.
        """
        self._content_server = content_server

    @property
    def uses_model(self) -> bool:
        return self.predictor.uses_model

    def _segment_for(self, segment_id: str):
        if self._content_server is not None:
            return self._content_server.get_segment(segment_id)
        from traffic_sim.content.segment import Segment

        cfg = self.config
        midpoint = (cfg.segment_bitrate_mbps_min + cfg.segment_bitrate_mbps_max) / 2.0
        return Segment(segment_id=segment_id, size_mb=1.0, duration_sec=10.0, bitrate_mbps=midpoint)

    def __call__(self, segment_id: str, ues: list[UE], cell: Cell, config: SimConfig) -> DeliveryMode:
        return self.decide(segment_id, ues, cell, config).mode

    def decide(self, segment_id: str, ues: list[UE], cell: Cell, config: SimConfig) -> AIDecision:
        key = (cell.cell_id, segment_id)
        state = self._groups.setdefault(key, _GroupState())
        segment = self._segment_for(segment_id)

        features = extract_features(
            segment,
            ues,
            cell,
            config,
            prev_congestion=self._prev_congestion.get(cell.cell_id, 0.0),
            prev_broadcast_groups=self._prev_broadcast_groups.get(cell.cell_id, 0),
        )
        prediction = self.predictor.predict(features)
        weights = resolve_weights(features, config)

        u_unicast = compute_utility(
            DeliveryMode.UNICAST, prediction.outcomes.unicast, features, config, state.current_mode, weights
        )
        u_broadcast = compute_utility(
            DeliveryMode.BROADCAST, prediction.outcomes.broadcast, features, config, state.current_mode, weights
        )

        best, other = (
            (u_broadcast, u_unicast) if u_broadcast.total > u_unicast.total else (u_unicast, u_broadcast)
        )
        candidate = best.mode
        gap = abs(u_broadcast.total - u_unicast.total)
        confidence = min(gap / 0.25, 1.0)  # 0.25 utility apart = fully confident

        current = state.current_mode
        reason: dict[str, object] = {
            "candidate": candidate.value,
            "utility_gap": round(gap, 4),
            "winning_terms": best.explain(),
            "prediction_source": prediction.source,
        }

        final = candidate
        if current is None:
            reason["action"] = "initial_assignment"
        elif candidate == current:
            reason["action"] = "hold_already_optimal"
        else:
            outcome_by_mode = {
                DeliveryMode.UNICAST: prediction.outcomes.unicast,
                DeliveryMode.BROADCAST: prediction.outcomes.broadcast,
            }
            final, hold_reason = self._apply_safety_gates(
                candidate,
                current,
                features,
                gap,
                confidence,
                config,
                state,
                outcome_current=outcome_by_mode[current],
                outcome_candidate=outcome_by_mode[candidate],
            )
            reason["action"] = hold_reason

        switched = current is not None and final != current
        if switched:
            state.last_switch_time_s = self._sim_time_s
        state.current_mode = final

        decision = AIDecision(
            mode=final,
            confidence=confidence,
            reason=reason,
            utility_unicast=u_unicast.total,
            utility_broadcast=u_broadcast.total,
            predicted_unicast_mbps=prediction.outcomes.unicast.achievable_mbps,
            predicted_broadcast_mbps=prediction.outcomes.broadcast.achievable_mbps,
            prediction_source=prediction.source,
            inference_latency_ms=prediction.inference_latency_ms,
            switched=switched,
        )
        self._tick_decisions.append(decision)
        return decision

    def _apply_safety_gates(
        self,
        candidate,
        current,
        features,
        gap,
        confidence,
        config,
        state,
        outcome_current,
        outcome_candidate,
    ) -> tuple[DeliveryMode, str]:
        """Return the mode we are actually allowed to use, and why.

        The gates exist to stop needless churn. But a gate must never trap
        the agent in a mode that is actively failing: if the current mode
        is predicted to stall playback and the alternative is not, holding
        is the more dangerous choice. That case escapes every gate except
        cooldown.
        """
        # Escape hatch. Note we compare DELIVERED CONTENT, not the stall
        # flags: once a buffer has hit zero every mode "would stall" by the
        # strict definition, so stall flags alone can never signal a way
        # out. Delivering meaningfully more content is what recovery
        # actually looks like.
        rescue = outcome_current.would_stall and (
            outcome_candidate.delivered_seconds > outcome_current.delivered_seconds * 1.05
        )
        if rescue:
            last = state.last_switch_time_s
            if last is not None and (self._sim_time_s - last) < config.cooldown_period_s:
                return current, "hold_cooldown"
            return candidate, "switch_escape_stall"

        # (a) Buffer safety — a switch costs transition time the worst-off
        #     UE may not be able to absorb. Only applies when staying put
        #     is actually survivable; see `rescue` above.
        needed = config.transition_time_seconds + config.safety_margin_seconds
        if features.buffer_min_s < needed:
            return current, "hold_buffer_unsafe"

        # (b) Confidence — the two options are effectively tied.
        if confidence < config.min_decision_confidence:
            return current, "hold_low_confidence"

        # (c) Hysteresis — real but too small a gain to justify churn.
        if gap <= config.hysteresis_margin:
            return current, "hold_within_hysteresis"

        # (d) Cooldown — this group switched too recently.
        last = state.last_switch_time_s
        if last is not None and (self._sim_time_s - last) < config.cooldown_period_s:
            return current, "hold_cooldown"

        return candidate, "switch"

    def current_tick_stats(self) -> TickStats:
        """Decisions made since the last `observe()`, for the KPI engine.

        Read before `observe()` clears them, so the numbers belong to the
        tick being recorded rather than the previous one.
        """
        latencies = [d.inference_latency_ms for d in self._tick_decisions]
        return TickStats(
            switch_count=sum(1 for d in self._tick_decisions if d.switched),
            mean_inference_latency_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
            decisions=list(self._tick_decisions),
        )

    def observe(self, cells, delivery_results, record) -> None:
        """Closed-loop feedback, called by the Simulator after each tick.

        The action the agent chose has now changed the network. Two things
        carry forward into the next decision: the congestion each cell
        actually experienced (a feature), and the simulation clock (which
        drives cooldown).
        """
        for cell in cells:
            self._prev_congestion[cell.cell_id] = cell.congestion()

        # How busy the broadcast carrier actually got, per cell. Next tick
        # this tells the agent whether adding another shared stream would
        # thin out the ones already there.
        # NOTE (measured limitation): every group reads this same number
        # and reacts to it identically, so groups tend to leave a busy
        # carrier together, which makes the carrier look free, which draws
        # them back. Exponentially smoothing this signal was tried and
        # measurably reduced churn (99 -> 79 switches) but slightly hurt
        # continuity and savings, so it was not kept. The oscillation is a
        # coordination problem between independent per-group decisions,
        # not a smoothing problem. See README for the numbers.
        broadcast_counts: dict[str, int] = {cell.cell_id: 0 for cell in cells}
        for result in delivery_results:
            if result.mode == "broadcast":
                broadcast_counts[result.cell_id] = broadcast_counts.get(result.cell_id, 0) + 1
        self._prev_broadcast_groups = broadcast_counts

        self._tick_decisions = []
        self._sim_time_s = getattr(
            record, "sim_time_s", self._sim_time_s + self.config.simulation_timestep_s
        )
