"""PCF: the policy brain (Guide Section 4, 16.4).

PCF stores and enforces rules — service priority, confidence thresholds,
buffer safety — independently of the AI agent that made the recommendation
it is reviewing. It runs last in the decision pipeline, immediately after
MBOS (`traffic_sim.mbos.controller`), and is the final gate before a mode
is handed to delivery:

    ai agent (predict -> utility -> safety gates) -> MBOS -> PCF -> final

Two rules, deliberately independent of `ai.agent._apply_safety_gates`
rather than a re-statement of it:

  1. Priority-scaled buffer margin. The agent's own buffer-safety gate
     uses ONE fixed margin for every group, regardless of content. PCF
     requires a LARGER margin for higher `Segment.service_priority`
     content before permitting any mode change this tick — delay-sensitive
     traffic gets a stronger continuity guarantee than delay-tolerant
     traffic, per Guide Section 10.5. A violation reverts to the
     incumbent (current) mode rather than blocking output entirely.

     This rule is PREVENTIVE, not absolute: it only applies while there
     is still some buffer left to protect. If a group's buffer has
     already hit zero, the stream has already failed, and the agent's own
     `switch_escape_stall` gate (traffic_sim.ai.agent) exists specifically
     to let it recover — vetoing that rescue here would recreate the
     buffer-safety deadlock fixed earlier in this project (a failing mode
     traps buffers at zero, and zero buffer then blocks the only way out).
     So a buffer of exactly zero is exempt from this rule; PCF still
     enforces it for any positive-but-insufficient buffer.
  2. Confidence floor, re-checked independently. The agent already checks
     `confidence >= min_decision_confidence` before it will switch, but
     PCF re-validates the same floor on whatever MBOS hands it rather
     than trusting the upstream result — ordinary defense-in-depth for a
     policy layer, and it means a future bug in the agent's own gate
     would still be caught here.

PCF does not touch the radio/delivery/utility math and does not decide
which mode is *better* — only whether the proposed mode is *allowed*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.config import SimConfig
from traffic_sim.simulation.user import DeliveryMode

PRIORITY_BUFFER_MARGIN = "priority_buffer_margin"
CONFIDENCE_FLOOR = "confidence_floor"


@dataclass
class PolicyEvaluation:
    """The outcome of running one decision through PCF."""

    final_mode: DeliveryMode
    compliant: bool
    violated_rules: list[str] = field(default_factory=list)

    def explain(self) -> dict[str, object]:
        return {
            "compliant": self.compliant,
            "violated_rules": list(self.violated_rules),
        }


class PCFPolicy:
    """Simulated Policy Control Function. One instance per AIAgent."""

    def __init__(self, config: SimConfig):
        self._config = config

    @property
    def reporting_interval_s(self) -> float:
        """How often the API layer should publish full decision detail.

        Exposed here (not re-implemented) because PCF is the guide's
        designated home for reporting-interval policy (Section 4); the
        value itself is `config.reporting_interval_s`, already used
        elsewhere in the simulator for telemetry cadence.
        """
        return self._config.reporting_interval_s

    def enforce(
        self,
        mode: DeliveryMode,
        current_mode: DeliveryMode | None,
        service_priority: int,
        buffer_min_s: float,
        confidence: float,
    ) -> PolicyEvaluation:
        cfg = self._config
        violated: list[str] = []
        final_mode = mode

        # Rule 1 — priority-scaled buffer margin. Only relevant when a
        # change is actually being proposed; holding the current mode
        # never needs a transition, so there is nothing to protect against.
        if current_mode is not None and mode != current_mode and buffer_min_s > 0:
            base_margin = cfg.transition_time_seconds + cfg.safety_margin_seconds
            required_margin = base_margin
            if service_priority >= cfg.pcf_priority_threshold:
                required_margin = base_margin * cfg.pcf_priority_margin_multiplier
            if buffer_min_s < required_margin:
                violated.append(PRIORITY_BUFFER_MARGIN)
                final_mode = current_mode

        # Rule 2 — confidence floor, re-checked independently of the agent.
        if confidence < cfg.min_decision_confidence:
            violated.append(CONFIDENCE_FLOOR)
            final_mode = current_mode if current_mode is not None else final_mode

        return PolicyEvaluation(final_mode=final_mode, compliant=not violated, violated_rules=violated)
