"""What would happen under each delivery mode — computed, not delivered.

This is the counterfactual evaluator. It answers "if we sent this group
by unicast, and if we sent it by broadcast, what would each look like?"
using the exact same pure functions the real delivery path uses
(`network.throughput`), so a label produced here matches what
`delivery.unicast` / `delivery.broadcast` would actually do.

It never calls `deliver()`, because that mutates UE buffers. Nothing in
this module changes simulator state.

Two consumers:
  * `ai.dataset` — to manufacture ground-truth training labels, which is
    legitimate because training happens offline.
  * `ai.predict` — as the analytical fallback when no trained model is
    available, so the agent still works out of the box.

A real network could not do this: it cannot try both modes and keep the
better one. That is precisely why the trained model exists — at decision
time it estimates these outcomes from telemetry alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.config import SimConfig
from traffic_sim.ai.feature_engineering import GroupFeatures
from traffic_sim.network.throughput import (
    broadcast_achievable_mbps,
    delivered_seconds_of_content,
    unicast_achievable_mbps,
)


@dataclass
class ModeOutcome:
    """Predicted or computed consequence of choosing one delivery mode."""

    achievable_mbps: float
    delivered_seconds: float
    # Buffer level the worst-off UE would hold after this tick.
    projected_min_buffer_s: float
    # True if that UE would run dry, i.e. playback would stall.
    would_stall: bool


@dataclass
class OutcomePair:
    unicast: ModeOutcome
    broadcast: ModeOutcome


def _to_outcome(
    achievable_mbps: float, features: GroupFeatures, config: SimConfig, dt: float
) -> ModeOutcome:
    delivered = delivered_seconds_of_content(achievable_mbps, features.segment_bitrate_mbps, dt)
    consumed = config.playback_consumption_rate * dt
    projected = max(0.0, features.buffer_min_s + delivered - consumed)
    return ModeOutcome(
        achievable_mbps=achievable_mbps,
        delivered_seconds=delivered,
        projected_min_buffer_s=projected,
        would_stall=projected <= 0.0,
    )


def evaluate_modes(features: GroupFeatures, config: SimConfig, dt: float | None = None) -> OutcomePair:
    """Compute the true outcome of each mode for one group, side-effect free."""
    dt = dt if dt is not None else config.simulation_timestep_s

    unicast_rate = unicast_achievable_mbps(
        shannon_mbps=features.shannon_unicast_mean_mbps,
        unicast_users_in_cell=features.unicast_sharing_users,
        cell_capacity_mbps=features.values["cell_capacity_mbps"],
        segment_bitrate_mbps=features.segment_bitrate_mbps,
    )
    broadcast_rate = broadcast_achievable_mbps(
        shannon_mbps_at_worst_sinr=features.shannon_broadcast_worst_mbps,
        broadcast_capacity_mbps=features.values["broadcast_capacity_mbps"],
        segment_bitrate_mbps=features.segment_bitrate_mbps,
        broadcast_groups_in_cell=features.broadcast_sharing_groups,
    )

    return OutcomePair(
        unicast=_to_outcome(unicast_rate, features, config, dt),
        broadcast=_to_outcome(broadcast_rate, features, config, dt),
    )
