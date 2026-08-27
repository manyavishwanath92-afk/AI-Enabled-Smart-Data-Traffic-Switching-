"""Cell-level broadcast admission control.

The problem this fixes: previously every (cell, segment) group decided
UNICAST vs BROADCAST independently, reacting to the same
"how contended is the broadcast carrier" reading. That made groups move
in lockstep — a busy carrier makes everyone leave broadcast at once,
which makes the carrier look free, which makes everyone pile back on —
an oscillation measured at ~99 switches/run under a constrained carrier.

The fix: decide once per cell, not once per group. Every group in a cell
is ranked by how much it would gain from broadcast, then admitted to the
shared carrier highest-gain-first, stopping (per candidate) as soon as
admitting it would leave the carrier unable to sustain the bitrate of
every group already admitted. Groups that don't fit fall back to
unicast. Because this whole decision is made once from one consistent
snapshot instead of many groups reacting independently and repeatedly,
there is no herd signal left to oscillate around.

This module only ranks and admits — it does not call XGBoost, does not
compute utility, and does not decide safety. `ai.agent.AIAgent` builds
the `GroupPlan`s (using the existing predictor and utility function
unchanged) and still applies the existing buffer/confidence/hysteresis/
cooldown/stall-escape gates on top of whatever this module decides.
"""

from __future__ import annotations

from dataclasses import dataclass

from traffic_sim.ai.feature_engineering import GroupFeatures
from traffic_sim.ai.outcomes import OutcomePair
from traffic_sim.ai.utility import UtilityBreakdown
from traffic_sim.simulation.user import DeliveryMode

GroupKey = tuple[str, str]  # (cell_id, segment_id)


@dataclass
class GroupPlan:
    """Everything computed for one group during cell-level planning.

    Cached so `AIAgent.decide()` does not repeat the XGBoost/utility work
    prepare_tick() already did for this group this tick.
    """

    key: GroupKey
    features: GroupFeatures
    outcomes: OutcomePair
    utility_unicast: UtilityBreakdown
    utility_broadcast: UtilityBreakdown
    broadcast_benefit: float
    prediction_source: str
    inference_latency_ms: float
    # The group's mode BEFORE this tick's decision — None on its first
    # ever decision. Used only to break admission ties in favour of
    # streams already on the carrier; see rank_by_broadcast_benefit.
    current_mode: DeliveryMode | None = None


def rank_by_broadcast_benefit(plans: list[GroupPlan], stickiness: float = 0.0) -> list[GroupPlan]:
    """Highest broadcast_benefit first, with a small tie-break toward
    groups already broadcasting.

    Without the tie-break, a cell with several similarly-ranked candidates
    reshuffles its admitted set every time the exact ranking order wobbles
    by a hair — e.g. one group's SINR drifts slightly this tick — even
    though nothing about the actual contention changed. `stickiness`
    (the caller passes config.hysteresis_margin — no new parameter) adds a
    small bonus to a group's benefit only for RANKING purposes if it is
    already on broadcast, so a newcomer must be genuinely better, not just
    marginally luckier this tick, to bump it. The real admission-capacity
    check afterwards is unaffected by this bonus — it only reorders who
    gets first refusal on a seat.
    """
    def sort_key(plan: GroupPlan) -> float:
        bonus = stickiness if plan.current_mode == DeliveryMode.BROADCAST else 0.0
        return plan.broadcast_benefit + bonus

    return sorted(plans, key=sort_key, reverse=True)


def _group_fits_on_shared_carrier(features: GroupFeatures, groups_sharing: int) -> bool:
    """Would this group's bitrate still be met if `groups_sharing` streams
    split the broadcast carrier evenly?

    Mirrors the real delivery-time formula
    (network.throughput.broadcast_achievable_mbps): both the carrier's
    raw capacity AND this group's own worst-case Shannon capacity are
    divided by the number of concurrent streams, and both must still
    cover the segment's bitrate.
    """
    if groups_sharing <= 0:
        return True
    bitrate = features.segment_bitrate_mbps
    capacity_share = features.values["broadcast_capacity_mbps"] / groups_sharing
    radio_share = features.shannon_broadcast_worst_mbps / groups_sharing
    return capacity_share >= bitrate and radio_share >= bitrate


def admit_to_broadcast(plans: list[GroupPlan], stickiness: float = 0.0) -> set[GroupKey]:
    """Greedily admit groups to one shared broadcast carrier.

    Only groups whose utility actually prefers broadcast (benefit > 0)
    are candidates — a group utility already scores as unicast-or-tied
    gains nothing from a broadcast seat. Candidates are offered a seat
    highest-benefit-first (with `stickiness` — see rank_by_broadcast_benefit
    — favouring groups already on broadcast in near-ties); a candidate
    that would leave the carrier unable to sustain the admitted set
    (itself included) is skipped, not treated as a hard stop, so a later,
    smaller-bitrate candidate can still fit in whatever headroom remains.
    """
    ranked = rank_by_broadcast_benefit(
        [plan for plan in plans if plan.broadcast_benefit > 0], stickiness=stickiness
    )

    admitted: list[GroupPlan] = []
    admitted_keys: set[GroupKey] = set()

    for candidate in ranked:
        trial = admitted + [candidate]
        groups_sharing = len(trial)
        if all(_group_fits_on_shared_carrier(plan.features, groups_sharing) for plan in trial):
            admitted = trial
            admitted_keys.add(candidate.key)

    return admitted_keys
