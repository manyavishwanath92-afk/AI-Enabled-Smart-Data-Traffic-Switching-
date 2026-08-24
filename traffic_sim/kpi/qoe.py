"""Quality of Experience (Guide Section 4, 11): a transparent weighted combination.

No ML — a plain, explainable weighted sum of buffer health, radio-quality
continuity, and starvation avoidance, each normalized to [0, 1].
"""

from __future__ import annotations

from traffic_sim.simulation.user import QoEState, UE

DEFAULT_WEIGHTS = {"buffer": 0.5, "continuity": 0.3, "no_starvation": 0.2}


def compute_qoe(
    ue: UE,
    sinr_threshold_linear: float,
    max_buffer_seconds: float = 30.0,
    weights: dict[str, float] | None = None,
) -> float:
    weights = weights or DEFAULT_WEIGHTS
    buffer_score = min(ue.buffer_seconds / max_buffer_seconds, 1.0) if max_buffer_seconds > 0 else 0.0
    continuity_score = 1.0 if (ue.sinr_linear is not None and ue.sinr_linear >= sinr_threshold_linear) else 0.0
    starvation_score = 0.0 if ue.buffer_seconds <= 0.0 else 1.0

    qoe = (
        weights.get("buffer", 0.0) * buffer_score
        + weights.get("continuity", 0.0) * continuity_score
        + weights.get("no_starvation", 0.0) * starvation_score
    )
    qoe = max(0.0, min(1.0, qoe))

    if ue.buffer_seconds <= 0.0:
        ue.qoe_state = QoEState.STARVED
    elif qoe >= 0.7:
        ue.qoe_state = QoEState.GOOD
    else:
        ue.qoe_state = QoEState.DEGRADED

    return qoe
