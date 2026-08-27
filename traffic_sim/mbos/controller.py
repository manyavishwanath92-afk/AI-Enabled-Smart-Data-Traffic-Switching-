"""MBOS control layer: coordinates the AI's chosen mode with a delivery bearer.

The architecture document describes MBOS as the central entity
coordinating unicast, multicast and broadcast delivery across
heterogeneous networks (Guide Section 16.1). In this PoC it is a
simulated controller with one concrete, testable job: turn "the AI wants
UNICAST or BROADCAST for this group" into a coordinated instruction —
a target delivery bearer plus a stable session — that the data plane
(`traffic_sim.delivery`) can act on.

MBOS does not re-decide anything the AI agent already decided (that would
duplicate `ai.agent`/`ai.utility`), and it does not enforce policy (that
is PCF's job — see `traffic_sim.policy.pcf`, which runs immediately
after MBOS in `AIAgent.decide()`). Its two responsibilities are narrower
and telecom-flavoured:

  1. Session/path coordination: track which bearer each (cell, segment)
     group is currently on, across ticks, and flag a HANDOVER when that
     changes — this is what the guide calls session continuity.
  2. Coherence: never forward an invalid instruction to delivery. If a
     mode somehow isn't a real DeliveryMode, MBOS falls back to the safe
     default (UNICAST) rather than passing garbage downstream.

This is a simulated controller, not a standards-compliant MBOS
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from traffic_sim.simulation.user import DeliveryMode

GroupKey = tuple[str, str]  # (cell_id, segment_id)

CELLULAR_BEARER = "cellular"
BROADCAST_BEARER = "broadcast"


def bearer_for(mode: DeliveryMode) -> str:
    """Which physical bearer a delivery mode maps to. Exposed (not
    private) so callers can re-derive the bearer for a mode PCF later
    reverts to, without re-running full coordination."""
    return BROADCAST_BEARER if mode == DeliveryMode.BROADCAST else CELLULAR_BEARER


@dataclass
class MBOSCoordinationResult:
    """What MBOS decided to hand to the data plane, and why."""

    mode: DeliveryMode
    session_id: str
    target_bearer: str
    handover: bool
    corrected: bool  # True only if the input mode was invalid and MBOS had to fall back

    def explain(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "target_bearer": self.target_bearer,
            "handover": self.handover,
            "corrected": self.corrected,
        }


class MBOSController:
    """Simulated MBOS control plane. One instance per Simulator/AIAgent.

    `coordinate()` PROPOSES a bearer for the agent's chosen mode without
    committing it; `confirm()` COMMITS whatever mode actually survives
    PCF's policy check. Splitting the two matters because PCF can still
    revert the mode after MBOS has proposed a bearer for it — if
    `coordinate()` recorded state eagerly, a policy-reverted proposal
    would corrupt next tick's handover detection with a bearer change
    that never really happened.
    """

    def __init__(self) -> None:
        self._last_bearer: dict[GroupKey, str] = {}

    def coordinate(self, cell_id: str, segment_id: str, mode: DeliveryMode) -> MBOSCoordinationResult:
        key = (cell_id, segment_id)
        session_id = f"{cell_id}:{segment_id}"

        corrected = False
        if not isinstance(mode, DeliveryMode) or mode == DeliveryMode.NONE:
            mode = DeliveryMode.UNICAST
            corrected = True

        target_bearer = bearer_for(mode)
        previous_bearer = self._last_bearer.get(key)
        handover = previous_bearer is not None and previous_bearer != target_bearer

        return MBOSCoordinationResult(
            mode=mode,
            session_id=session_id,
            target_bearer=target_bearer,
            handover=handover,
            corrected=corrected,
        )

    def confirm(self, cell_id: str, segment_id: str, mode: DeliveryMode) -> None:
        """Commit the truly-final mode (after PCF) for next tick's handover
        detection. Must be called once per group per tick, after PCF."""
        self._last_bearer[(cell_id, segment_id)] = bearer_for(mode)
