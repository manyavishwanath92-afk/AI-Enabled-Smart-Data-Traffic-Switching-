"""The AI decision agent — the final say on UNICAST vs BROADCAST.

It is not the model, and it is not the utility function. Those answer
"what will happen" and "which outcome is better". The agent answers a
third question: "what are we actually allowed to do right now?"

Order of reasoning, per tick:

  0. (if the simulator supports it) COORDINATE across the whole cell
     first -> ai.cell_controller. Every group's telemetry, prediction and
     utility are computed together, ranked by how much each would gain
     from broadcast, and admitted to the shared broadcast carrier
     highest-gain-first while it can still sustain everyone admitted.
     This is what prevents groups from reacting to the same contention
     signal independently and moving in lockstep (measured at ~99
     switches/run before this existed — see README).

  Then, per (cell, segment) group:

  1. read telemetry            -> ai.feature_engineering  (cached by step 0)
  2. predict both outcomes     -> ai.predict               (cached by step 0)
  3. score both outcomes       -> ai.utility                (cached by step 0)
  4. take the coordinated candidate from step 0 (or, if step 0 did not
     run for this tick — e.g. direct calls to `decide()` in a test — fall
     back to this group's own utility argmax, exactly as before
     coordination existed)
  5. refuse to act if unsafe:
       a. buffer safety - too little margin to survive a switch
       b. low confidence - the two options are nearly tied
       c. hysteresis    - the gain is real but too small to be worth churn
       d. cooldown      - this group switched too recently
  6. MBOS   -> mbos.controller.MBOSController: coordinate a session/bearer
     for the resulting mode (session id, target bearer, handover flag)
  7. PCF    -> policy.pcf.PCFPolicy: independently re-validate the
     MBOS-coordinated mode against policy rules (priority-scaled buffer
     margin, confidence floor), able to revert to the incumbent mode

Steps 5a-5d are why the agent exists. Without them a marginally better
score would flip a group's delivery mode every tick, which costs more in
churn than it gains in throughput. Coordination (step 0) and these gates
solve two different causes of the same symptom: coordination stops
groups from herding on a shared signal; the gates stop a single group
from flip-flopping on its own marginal, noisy re-evaluations. Steps 6-7
are deliberately separate from the agent's own reasoning: MBOS's job is
session/path coordination, not decision-making, and PCF's job is policy
enforcement independent of (and after) the agent that proposed the mode —
neither duplicates `_apply_safety_gates`.

`AIAgent` is a callable object with the exact `ModeSelectorFn` signature,
so it drops into the simulator's existing seam with no interface change.
It also exposes `prepare_tick()` (coordination, called once per tick
before any group is decided) and `observe()` (closed-loop feedback,
called once per tick after delivery) — both are optional hooks the
Simulator calls only if they exist, so plain-rule selectors are
completely unaffected. MBOS and PCF are internal to the AI pipeline
specifically (owned by `AIAgent`, not the Simulator), so fixed-rule
baselines (`always_unicast`, `rule_by_group_size`, ...) remain
unpoliced, "raw" comparison points for `traffic_sim.experiments.compare`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from config.config import SimConfig
from traffic_sim.ai.cell_controller import GroupPlan, admit_to_broadcast
from traffic_sim.ai.feature_engineering import extract_features
from traffic_sim.ai.predict import OutcomePredictor
from traffic_sim.ai.utility import UtilityBreakdown, compute_utility, resolve_weights
from traffic_sim.content.content_server import ContentServer
from traffic_sim.mbos.controller import MBOSController, bearer_for
from traffic_sim.policy.pcf import PCFPolicy
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import DeliveryMode, UE


@dataclass
class AIDecision:
    """One explainable decision."""

    mode: DeliveryMode
    confidence: float
    reason: dict[str, object]
    cell_id: str
    segment_id: str
    utility_unicast: float
    utility_broadcast: float
    predicted_unicast_mbps: float
    predicted_broadcast_mbps: float
    prediction_source: str
    inference_latency_ms: float
    switched: bool
    # The mode this group was in immediately before this decision (None on
    # a group's very first decision). Carried out here because it can't be
    # reconstructed afterward — `_GroupState.current_mode` is overwritten
    # with the new final mode by the time this decision returns.
    previous_mode: DeliveryMode | None
    # MBOS — session/bearer coordination (traffic_sim.mbos.controller).
    session_id: str
    target_bearer: str
    handover: bool
    # PCF — independent policy enforcement (traffic_sim.policy.pcf). May
    # have overridden `mode` back to the incumbent; violated_rules is
    # empty when compliant.
    pcf_compliant: bool
    pcf_violated_rules: list[str]


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


@dataclass
class DecisionRecord:
    """One dashboard-friendly, persisted row of decision history.

    Deliberately lighter than `AIDecision`: plain JSON-serialisable values
    (modes as their `.value` strings, not enums), no nested `reason` dict,
    no `DeliveryResult`. `selected_mode`/`mbos_bearer` are read straight off
    the already-reconciled post-MBOS-post-PCF `AIDecision` fields, so this
    is guaranteed to reflect the FINAL delivered mode, never the agent's
    pre-PCF candidate.
    """

    tick: int
    sim_time_s: float
    cell_id: str
    group_id: str  # segment id
    previous_mode: str | None
    selected_mode: str
    utility_unicast: float
    utility_broadcast: float
    broadcast_benefit: float
    confidence: float
    decision_reason: str
    mbos_bearer: str
    pcf_result: str
    inference_latency_ms: float


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
        self.mbos = MBOSController()
        self.pcf = PCFPolicy(config)
        self._groups: dict[tuple[str, str], _GroupState] = {}
        self._prev_congestion: dict[str, float] = {}
        self._prev_broadcast_groups: dict[str, int] = {}
        self._sim_time_s = 0.0
        self._tick_decisions: list[AIDecision] = []
        # Bounded, dashboard-friendly decision history — see DecisionRecord.
        # Populated in observe() (which has the tick's real KPIRecord),
        # never grows past decision_history_size.
        self._decision_history: deque[DecisionRecord] = deque(maxlen=config.decision_history_size)
        # Populated by prepare_tick(); looked up by decide(). Empty when
        # prepare_tick() was not called this tick, in which case decide()
        # falls back to computing everything for itself (see decide()).
        self._tick_plans: dict[tuple[str, str], GroupPlan] = {}
        self._coordinated_modes: dict[tuple[str, str], DeliveryMode] = {}

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

    def _build_plan(
        self, key: tuple[str, str], segment_id: str, ues: list[UE], cell: Cell, config: SimConfig
    ) -> GroupPlan:
        """Run telemetry -> prediction -> utility for one group.

        Shared by `prepare_tick()` (called once per group, for every group
        in a cell) and `decide()`'s fallback path (called for a single
        group when no coordination pass has run this tick), so the two
        paths can never compute this differently.
        """
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
        state = self._groups.setdefault(key, _GroupState())

        u_unicast = compute_utility(
            DeliveryMode.UNICAST, prediction.outcomes.unicast, features, config, state.current_mode, weights
        )
        u_broadcast = compute_utility(
            DeliveryMode.BROADCAST, prediction.outcomes.broadcast, features, config, state.current_mode, weights
        )

        return GroupPlan(
            key=key,
            features=features,
            outcomes=prediction.outcomes,
            utility_unicast=u_unicast,
            utility_broadcast=u_broadcast,
            broadcast_benefit=max(0.0, u_broadcast.total - u_unicast.total),
            prediction_source=prediction.source,
            inference_latency_ms=prediction.inference_latency_ms,
            current_mode=state.current_mode,
        )

    def prepare_tick(
        self,
        groups: dict[tuple[str, str], list[UE]],
        cells_by_id: dict[str, Cell],
        config: SimConfig,
    ) -> None:
        """Coordinate one broadcast-admission decision per cell.

        Called once per tick by the Simulator, before any `decide()` call
        for that tick (an optional hook — see module docstring). Builds
        every group's plan up front, ranks each cell's candidates by
        `broadcast_benefit`, and admits them to the shared broadcast
        carrier while it can still sustain everyone admitted
        (`ai.cell_controller.admit_to_broadcast`). `decide()` then simply
        looks up the resulting mode instead of comparing utilities alone,
        which is what stops every group from reacting to the same
        contention signal independently.
        """
        self._tick_plans = {}
        self._coordinated_modes = {}

        plans_by_cell: dict[str, list[GroupPlan]] = {}
        for (cell_id, segment_id), ues in groups.items():
            key = (cell_id, segment_id)
            plan = self._build_plan(key, segment_id, ues, cells_by_id[cell_id], config)
            self._tick_plans[key] = plan
            plans_by_cell.setdefault(cell_id, []).append(plan)

        for cell_plans in plans_by_cell.values():
            admitted_keys = admit_to_broadcast(cell_plans, stickiness=config.hysteresis_margin)
            for plan in cell_plans:
                self._coordinated_modes[plan.key] = (
                    DeliveryMode.BROADCAST if plan.key in admitted_keys else DeliveryMode.UNICAST
                )

    def decide(self, segment_id: str, ues: list[UE], cell: Cell, config: SimConfig) -> AIDecision:
        key = (cell.cell_id, segment_id)
        state = self._groups.setdefault(key, _GroupState())

        plan = self._tick_plans.get(key)
        coordinated = key in self._coordinated_modes
        if plan is None:
            # No prepare_tick() this tick (e.g. a test calling decide()
            # directly) — fall back to this group's own utility argmax,
            # exactly the pre-coordination behaviour.
            plan = self._build_plan(key, segment_id, ues, cell, config)

        u_unicast = plan.utility_unicast
        u_broadcast = plan.utility_broadcast
        prediction_source = plan.prediction_source
        inference_latency_ms = plan.inference_latency_ms
        outcome_by_mode = {DeliveryMode.UNICAST: plan.outcomes.unicast, DeliveryMode.BROADCAST: plan.outcomes.broadcast}

        if coordinated:
            candidate = self._coordinated_modes[key]
            best = u_broadcast if candidate == DeliveryMode.BROADCAST else u_unicast
        else:
            best = u_broadcast if u_broadcast.total > u_unicast.total else u_unicast
            candidate = best.mode

        gap = abs(u_broadcast.total - u_unicast.total)
        confidence = min(gap / 0.25, 1.0)  # 0.25 utility apart = fully confident

        current = state.current_mode
        reason: dict[str, object] = {
            "candidate": candidate.value,
            "utility_gap": round(gap, 4),
            "broadcast_benefit": round(plan.broadcast_benefit, 4),
            "winning_terms": best.explain(),
            "prediction_source": prediction_source,
            "coordination": "cell_admission" if coordinated else "independent",
        }

        final = candidate
        if current is None:
            reason["action"] = "initial_assignment"
        elif candidate == current:
            reason["action"] = "hold_already_optimal"
        else:
            final, hold_reason = self._apply_safety_gates(
                candidate,
                current,
                plan.features,
                gap,
                confidence,
                config,
                state,
                outcome_current=outcome_by_mode[current],
                outcome_candidate=outcome_by_mode[candidate],
            )
            reason["action"] = hold_reason

        # MBOS — coordinate a session/bearer for whatever the agent's own
        # reasoning settled on, regardless of whether that was a hold or a
        # switch (session/handover tracking must run every tick either way).
        mbos_result = self.mbos.coordinate(cell_id=key[0], segment_id=key[1], mode=final)

        # PCF — independent policy re-validation of the MBOS-coordinated
        # mode. May revert back to the incumbent mode; that reversion is
        # what actually gets committed below, not the agent's own proposal.
        pcf_result = self.pcf.enforce(
            mode=mbos_result.mode,
            current_mode=current,
            service_priority=int(plan.features.values["service_priority"]),
            buffer_min_s=plan.features.buffer_min_s,
            confidence=confidence,
        )
        final = pcf_result.final_mode

        # If PCF reverted the mode, MBOS's proposed bearer/handover no
        # longer describe what's actually being delivered — reconcile
        # them against the true final mode rather than report a bearer
        # switch that didn't happen.
        reported_bearer = mbos_result.target_bearer
        reported_handover = mbos_result.handover
        if final != mbos_result.mode:
            reported_bearer = bearer_for(final)
            reported_handover = False

        reason["mbos"] = mbos_result.explain()
        reason["pcf"] = pcf_result.explain()
        if not pcf_result.compliant:
            reason["action"] = "hold_pcf_policy_violation"
            reason["mbos"]["target_bearer"] = reported_bearer
            reason["mbos"]["handover"] = reported_handover

        self.mbos.confirm(cell_id=key[0], segment_id=key[1], mode=final)

        switched = current is not None and final != current
        if switched:
            state.last_switch_time_s = self._sim_time_s
        state.current_mode = final

        decision = AIDecision(
            mode=final,
            confidence=confidence,
            reason=reason,
            cell_id=key[0],
            segment_id=key[1],
            utility_unicast=u_unicast.total,
            utility_broadcast=u_broadcast.total,
            predicted_unicast_mbps=plan.outcomes.unicast.achievable_mbps,
            predicted_broadcast_mbps=plan.outcomes.broadcast.achievable_mbps,
            prediction_source=prediction_source,
            inference_latency_ms=inference_latency_ms,
            switched=switched,
            previous_mode=current,
            session_id=mbos_result.session_id,
            target_bearer=reported_bearer,
            handover=reported_handover,
            pcf_compliant=pcf_result.compliant,
            pcf_violated_rules=pcf_result.violated_rules,
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

    def decision_history(self, limit: int | None = None) -> list[DecisionRecord]:
        """Recent decision rows for a dashboard/API layer.

        Unlike `current_tick_stats().decisions`, this survives past the end
        of a run — it's populated in `observe()` each tick and bounded to
        `config.decision_history_size` rows (oldest dropped first).
        """
        records = list(self._decision_history)
        return records[-limit:] if limit else records

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

        # Turn this tick's decisions into bounded, dashboard-friendly
        # history rows before they're cleared below. `record` (this tick's
        # real KPIRecord) is only available here, not inside decide() —
        # that's why history is built in observe(), not decide().
        for decision in self._tick_decisions:
            pcf_result = (
                "compliant"
                if decision.pcf_compliant
                else "reverted:" + ",".join(decision.pcf_violated_rules)
            )
            self._decision_history.append(
                DecisionRecord(
                    tick=record.tick,
                    sim_time_s=record.sim_time_s,
                    cell_id=decision.cell_id,
                    group_id=decision.segment_id,
                    previous_mode=decision.previous_mode.value if decision.previous_mode else None,
                    selected_mode=decision.mode.value,
                    utility_unicast=decision.utility_unicast,
                    utility_broadcast=decision.utility_broadcast,
                    broadcast_benefit=decision.reason.get("broadcast_benefit", 0.0),
                    confidence=decision.confidence,
                    decision_reason=decision.reason.get("action", ""),
                    mbos_bearer=decision.target_bearer,
                    pcf_result=pcf_result,
                    inference_latency_ms=decision.inference_latency_ms,
                )
            )

        self._tick_decisions = []
        # Belt-and-braces: prepare_tick() already resets these at the start
        # of the next tick, but clearing here too means a tick where it is
        # NOT called can't accidentally reuse a stale coordinated plan.
        self._tick_plans = {}
        self._coordinated_modes = {}
        self._sim_time_s = getattr(
            record, "sim_time_s", self._sim_time_s + self.config.simulation_timestep_s
        )
