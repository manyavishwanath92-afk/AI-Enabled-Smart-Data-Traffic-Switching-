"""Serializers turning simulator objects into plain JSON-friendly dicts.

Written out explicitly rather than leaning on a generic encoder so the wire
format is a deliberate, stable contract the frontend can rely on — field names
here match docs/FRONTEND_API_HANDOFF.md Part 3 exactly. Nothing in this module
computes a metric; it only reshapes values the simulator already produced.
"""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
from typing import Any

from traffic_sim.api.explain import explain_decision, explain_record
from traffic_sim.delivery.result import DeliveryResult
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import UE


def _enum_value(value: Any) -> Any:
    """`DeliveryMode.UNICAST -> "unicast"`, leaving None/plain values alone."""
    return value.value if isinstance(value, Enum) else value


def cell_to_dict(cell: Cell) -> dict[str, Any]:
    """One cell's live state, including its computed congestion.

    `congestion` is a method on `Cell`, not a stored field, so it would be
    missing from a naive dataclass dump — the dashboard's headline load
    indicator depends on it. `connected_ue_ids` is a set; it is sorted into a
    list purely so JSON can carry it and the ordering is stable across polls.
    """
    return {
        "cell_id": cell.cell_id,
        "center_x_km": cell.center_x_km,
        "center_y_km": cell.center_y_km,
        "maximum_capacity_mbps": cell.maximum_capacity_mbps,
        "broadcast_capacity_mbps": cell.broadcast_capacity_mbps,
        "current_load_mbps": cell.current_load_mbps,
        "congestion": cell.congestion(),
        "connected_ue_ids": sorted(cell.connected_ue_ids),
        "num_connected_ues": len(cell.connected_ue_ids),
    }


def ue_to_dict(ue: UE) -> dict[str, Any]:
    """One UE's live state, with every enum flattened to its string value."""
    return {
        "ue_id": ue.ue_id,
        "x_km": ue.x_km,
        "y_km": ue.y_km,
        "cell_id": ue.cell_id,
        "requested_segment": ue.requested_segment,
        "buffer_seconds": ue.buffer_seconds,
        "mobility_state": _enum_value(ue.mobility_state),
        "delivery_mode": _enum_value(ue.delivery_mode),
        "qoe_state": _enum_value(ue.qoe_state),
        "radio_state": _enum_value(ue.radio_state),
        "velocity_mps": ue.velocity_mps,
        "direction_rad": ue.direction_rad,
        "sinr_linear": ue.sinr_linear,
        "mobility_risk": ue.mobility_risk,
        "stall_ticks": ue.stall_ticks,
        "playback_ticks": ue.playback_ticks,
        "rebuffering_events": ue.rebuffering_events,
    }


def decision_to_dict(decision: Any) -> dict[str, Any]:
    """One `AIDecision`, reduced to what a dashboard should show.

    Deliberately NOT the full record. The agent's internal chain — the
    per-term utility breakdown, the raw utility scores, the predicted rates,
    which predictor ran, MBOS/PCF internals — stays inside the backend, where
    it is still computed and still tested. What comes out here is the part a
    viewer can act on: the final mode, how sure the agent was, and one plain
    sentence saying why.

    `mode` is `AIDecision.mode`, i.e. the mode after MBOS coordination and
    PCF enforcement — never the agent's pre-policy candidate. Field names
    match `DecisionRecord` so the live and historical views line up.
    """
    selected_mode = _enum_value(decision.mode)
    previous_mode = _enum_value(decision.previous_mode)
    return {
        "cell_id": decision.cell_id,
        "group_id": decision.segment_id,
        "selected_mode": selected_mode,
        "previous_mode": previous_mode,
        "switched": decision.switched,
        "confidence": decision.confidence,
        "explanation": explain_decision(
            action=str(decision.reason.get("action", "")),
            selected_mode=selected_mode,
            previous_mode=previous_mode,
            violated_rules=list(decision.pcf_violated_rules),
        ),
        "bearer": decision.target_bearer,
        "policy_compliant": decision.pcf_compliant,
    }


def decision_record_to_dict(record: Any) -> dict[str, Any]:
    """One persisted `DecisionRecord`, plus its rendered explanation.

    Every stored field is kept as-is (the history contract predates this and
    charts may use the numbers); `explanation` is added so a history table can
    be read without decoding `decision_reason` codes on the frontend.
    """
    payload = asdict(record)
    payload["explanation"] = explain_record(payload)
    return payload


def delivery_result_to_dict(result: DeliveryResult) -> dict[str, Any]:
    """One group's `DeliveryResult` — plain dicts/lists throughout."""
    return asdict(result)
