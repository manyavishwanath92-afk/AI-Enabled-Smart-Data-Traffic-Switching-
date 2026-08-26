"""The utility model: turn predicted outcomes into one comparable score.

The prediction step says what will *happen*. This step says which
outcome we prefer, and why. Keeping them apart is what makes the system
explainable: every decision can be traced to named, numeric terms rather
than to an opaque classifier output.

    sustain = min(achievable_rate / segment_bitrate, 1)   # can it keep up?

    U = w_quality      * sustain
      + w_resource     * ResourceSaving    * sustain
      + w_popularity   * PopularityBenefit * sustain
      + w_buffer       * BufferSafety
      - w_mobility     * MobilityRisk      (group may disperse mid-stream)
      - w_switch       * SwitchCost        (churn is not free)

Every term is normalised to [0, 1] so the weights alone decide their
relative importance.

Note what `sustain` multiplies. Spectrum saved on a stream too slow to
watch is not a saving, and a shared copy nobody can decode is not a
benefit — so the efficiency terms are earned in proportion to whether
delivery actually works. Without this, broadcast banks its large resource
and popularity rewards even while starving its listeners, and wins on
paper regardless of the radio. Buffer safety is deliberately NOT scaled:
it is a hazard measure, not a benefit. Weights adapt to context (guide Section 14.4): when
the cell is congested, saving capacity matters more; when buffers are
low, safety matters more; when UEs are moving fast, committing a group
to a shared stream is riskier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.config import SimConfig
from traffic_sim.ai.feature_engineering import GroupFeatures
from traffic_sim.ai.outcomes import ModeOutcome
from traffic_sim.simulation.user import DeliveryMode

DEFAULT_WEIGHTS: dict[str, float] = {
    "quality": 0.35,
    "resource": 0.25,
    "buffer": 0.20,
    "popularity": 0.10,
    "mobility": 0.05,
    "switch": 0.05,
}


@dataclass
class UtilityBreakdown:
    """Every term that fed the score, so a decision can be explained."""

    mode: DeliveryMode
    total: float
    terms: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def explain(self) -> dict[str, float]:
        return {name: round(self.terms[name] * self.weights.get(name, 0.0), 4) for name in self.terms}


def resolve_weights(features: GroupFeatures, config: SimConfig) -> dict[str, float]:
    """Start from configured weights, then adapt them to the situation."""
    weights = dict(config.utility_weights or DEFAULT_WEIGHTS)

    congestion = max(features.values["congestion_estimate"], features.values["prev_congestion"])
    if congestion > 0.7:
        weights["resource"] = weights.get("resource", 0.0) * 1.5
    if features.buffer_min_s < config.transition_time_seconds + config.safety_margin_seconds:
        weights["buffer"] = weights.get("buffer", 0.0) * 2.0
    if features.values["mobility_risk_mean"] > 0.5:
        weights["mobility"] = weights.get("mobility", 0.0) * 2.0
    if features.values["service_priority"] >= 2:
        weights["quality"] = weights.get("quality", 0.0) * 1.5

    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {name: value / total for name, value in weights.items()}


def compute_utility(
    mode: DeliveryMode,
    outcome: ModeOutcome,
    features: GroupFeatures,
    config: SimConfig,
    current_mode: DeliveryMode | None,
    weights: dict[str, float] | None = None,
) -> UtilityBreakdown:
    weights = weights if weights is not None else resolve_weights(features, config)
    bitrate = max(features.segment_bitrate_mbps, 1e-9)

    # Can this mode keep the stream playing? 1.0 = full bitrate delivered.
    quality = min(outcome.achievable_mbps / bitrate, 1.0)

    # Unicast burns capacity per recipient; broadcast burns one copy.
    if mode == DeliveryMode.BROADCAST:
        resource_used = config.segment_resource_mbps
    else:
        resource_used = features.n_requesting * config.segment_resource_mbps
    capacity = max(features.values["cell_capacity_mbps"], 1e-9)
    resource_saving = 1.0 - min(resource_used / capacity, 1.0)

    # How much margin the worst-off UE would have after this tick.
    safe_level = config.transition_time_seconds + config.safety_margin_seconds
    buffer_safety = min(outcome.projected_min_buffer_s / max(safe_level, 1e-9), 1.0)
    if outcome.would_stall:
        buffer_safety = 0.0

    # One shared copy is worth more the more UEs share it.
    if mode == DeliveryMode.BROADCAST and features.n_requesting > 1:
        popularity_benefit = 1.0 - (1.0 / features.n_requesting)
    else:
        popularity_benefit = 0.0

    # Mobility only threatens a shared stream: a dispersing group breaks
    # broadcast, while unicast simply follows each UE.
    mobility_risk = features.values["mobility_risk_mean"] if mode == DeliveryMode.BROADCAST else 0.0

    switch_cost = 1.0 if (current_mode is not None and mode != current_mode) else 0.0

    # Efficiency is only worth what it delivers — see module docstring.
    earned_resource = resource_saving * quality
    earned_popularity = popularity_benefit * quality

    terms = {
        "quality": quality,
        "resource": earned_resource,
        "buffer": buffer_safety,
        "popularity": earned_popularity,
        "mobility": mobility_risk,
        "switch": switch_cost,
    }

    total = (
        weights.get("quality", 0.0) * quality
        + weights.get("resource", 0.0) * earned_resource
        + weights.get("buffer", 0.0) * buffer_safety
        + weights.get("popularity", 0.0) * earned_popularity
        - weights.get("mobility", 0.0) * mobility_risk
        - weights.get("switch", 0.0) * switch_cost
    )

    return UtilityBreakdown(mode=mode, total=total, terms=terms, weights=weights)
