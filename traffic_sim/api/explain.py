"""Plain-English rendering of a decision the AI agent already made.

This is presentation, not reasoning. The agent records *why* it settled on a
mode as a short code (`switch`, `hold_cooldown`, ...); this module turns that
code into one sentence a non-specialist can read. It adds no logic, no
thresholds and no judgement of its own, which is why it lives in the API
layer and `traffic_sim.ai` needs no changes to support it.

The dashboard deliberately shows only the final mode, the confidence, and one
of these sentences — not the internal prediction -> utility -> agent chain.
Those internals still exist and are still tested; they are simply not the
right thing to put in front of a dashboard viewer.
"""

from __future__ import annotations

# Why PCF overrode the agent, in the reader's terms rather than the rule's name.
_PCF_RULES = {
    "priority_buffer_margin": "there isn't enough buffered playback to safely switch this content",
    "confidence_floor": "the decision wasn't confident enough to act on",
}

# One sentence per reason code the agent emits (traffic_sim.ai.agent).
# `{mode}` is the mode actually in force after the decision; `{previous}` the
# one before it.
_TEMPLATES = {
    "initial_assignment": "Started on {mode} — first decision for this group.",
    "hold_already_optimal": "Staying on {mode} — it is still the best option.",
    "switch": "Switched to {mode} — it now delivers this content better.",
    "switch_escape_stall": "Switched to {mode} — playback was stalling on {previous}.",
    "hold_buffer_unsafe": "Staying on {mode} — buffers are too low to risk switching.",
    "hold_low_confidence": "Staying on {mode} — the two options are too close to call.",
    "hold_within_hysteresis": "Staying on {mode} — the gain from switching is too small to be worth it.",
    "hold_cooldown": "Staying on {mode} — it switched too recently to change again.",
}

_UNKNOWN = "Using {mode} — decision recorded as '{action}'."


def explain_decision(
    action: str,
    selected_mode: str,
    previous_mode: str | None = None,
    violated_rules: list[str] | None = None,
) -> str:
    """One human-readable sentence for a decision.

    `action` is the agent's reason code, `selected_mode` the FINAL mode after
    MBOS coordination and PCF enforcement.
    """
    if action == "hold_pcf_policy_violation":
        rules = violated_rules or []
        causes = [_PCF_RULES.get(rule, rule.replace("_", " ")) for rule in rules]
        because = "; ".join(causes) if causes else "a network policy blocked the change"
        return f"Staying on {selected_mode} — network policy blocked the change: {because}."

    template = _TEMPLATES.get(action)
    if template is None:
        # An unrecognised code must degrade to a readable sentence, never
        # break the dashboard: a serving layer has no business raising here.
        return _UNKNOWN.format(mode=selected_mode, action=action)

    return template.format(mode=selected_mode, previous=previous_mode or "the previous mode")


def explain_record(record: dict) -> str:
    """Same, for a persisted `DecisionRecord` already serialized to a dict.

    History rows carry PCF's outcome as `"compliant"` or
    `"reverted:<rule>[,<rule>]"` rather than a rule list, so unpack that back
    into rule names before rendering.
    """
    pcf_result = record.get("pcf_result", "")
    violated = pcf_result.split(":", 1)[1].split(",") if pcf_result.startswith("reverted:") else []
    return explain_decision(
        action=record.get("decision_reason", ""),
        selected_mode=record.get("selected_mode", "unknown"),
        previous_mode=record.get("previous_mode"),
        violated_rules=[rule for rule in violated if rule],
    )
