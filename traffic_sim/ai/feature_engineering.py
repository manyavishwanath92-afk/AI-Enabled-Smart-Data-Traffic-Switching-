"""Turn live network state into the feature vector the AI reasons over.

Every feature here must be knowable BEFORE delivery happens, because that
is when the decision is made. Nothing in this module calls `deliver()` or
mutates any UE, cell or segment — extracting features must never change
the thing being measured.

`FEATURE_NAMES` is the single source of truth for column order. Training
and inference both build their vectors through `features_to_vector`, so
the two can never silently disagree about which number means what.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from config.config import SimConfig
from traffic_sim.content.segment import Segment
from traffic_sim.network.radio import compute_ue_radio_state
from traffic_sim.network.resources import congestion as compute_congestion
from traffic_sim.network.resources import required_resource_unicast
from traffic_sim.network.sinr import sinr_to_db
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import UE

FEATURE_NAMES: list[str] = [
    "n_requesting",           # how many UEs want this segment here
    "n_active_in_cell",       # how busy the cell is overall
    "demand_ratio",           # n_requesting / n_active_in_cell
    "segment_bitrate_mbps",   # rate needed to play in real time
    "segment_popularity",     # share of all UEs wanting this segment
    "service_priority",       # 0 = delay-tolerant .. higher = sensitive
    "buffer_min_s",           # worst-off UE in the group
    "buffer_mean_s",
    "sinr_mean_db",
    "sinr_min_db",            # worst radio in the group (sizes broadcast)
    "shannon_mean_mbps",      # cellular-carrier capacity at mean SINR
    "shannon_min_mbps",
    "mobility_risk_mean",     # will the group disperse?
    "mobility_risk_max",
    "cell_capacity_mbps",
    "broadcast_capacity_mbps",
    "unicast_load_estimate_mbps",  # n_requesting * per-stream cost
    "congestion_estimate",         # that load vs cell capacity
    "prev_congestion",             # what the cell actually saw last tick
    "prev_broadcast_groups",       # how contended the broadcast carrier was
]


@dataclass
class GroupFeatures:
    """Named features plus the raw pieces the utility model also needs."""

    values: dict[str, float]
    # Carried alongside because the utility function needs them directly
    # and re-deriving them would risk drift.
    n_requesting: int
    segment_bitrate_mbps: float
    buffer_min_s: float
    shannon_unicast_mean_mbps: float
    shannon_broadcast_worst_mbps: float
    unicast_sharing_users: int
    # How many streams the broadcast carrier would be split across if this
    # group joined it. Estimated from the previous tick — the agent cannot
    # know what the other groups will choose this tick.
    broadcast_sharing_groups: int

    def vector(self) -> list[float]:
        return [self.values[name] for name in FEATURE_NAMES]


def extract_features(
    segment: Segment,
    ues: list[UE],
    cell: Cell,
    config: SimConfig,
    prev_congestion: float = 0.0,
    prev_broadcast_groups: int = 0,
) -> GroupFeatures:
    """Build the decision-time feature set for one (cell, segment) group.

    `unicast_sharing_users` is an ESTIMATE. At decision time we do not yet
    know how many UEs the cell will end up serving by unicast this tick —
    that depends on decisions not made yet, including this one. We assume
    the cell keeps serving its whole current population by unicast, which
    is the pessimistic case for unicast and therefore a safe basis for
    choosing to offload.
    """
    n_requesting = max(len(ues), 1)
    n_active = max(len(cell.connected_ue_ids), n_requesting)

    buffers = [ue.buffer_seconds for ue in ues] or [0.0]
    risks = [ue.mobility_risk if ue.mobility_risk is not None else 0.0 for ue in ues] or [0.0]

    # Radio state on each carrier. Pure reads — no UE is modified.
    cellular_states = [
        compute_ue_radio_state(ue, cell, config, n_active, config.cellular_bandwidth_hz) for ue in ues
    ]
    broadcast_states = [
        compute_ue_radio_state(ue, cell, config, n_active, config.broadcast_bandwidth_hz) for ue in ues
    ]

    sinrs = [s.sinr_linear for s in cellular_states] or [0.0]
    sinr_db = [sinr_to_db(s) if s > 0 else -100.0 for s in sinrs]
    shannon_cell = [s.shannon_mbps for s in cellular_states] or [0.0]
    shannon_bc = [s.shannon_mbps for s in broadcast_states] or [0.0]

    broadcast_capacity = cell.broadcast_capacity_mbps or cell.maximum_capacity_mbps
    unicast_load = required_resource_unicast(n_requesting, config.segment_resource_mbps)

    values = {
        "n_requesting": float(n_requesting),
        "n_active_in_cell": float(n_active),
        "demand_ratio": n_requesting / n_active,
        "segment_bitrate_mbps": float(segment.bitrate_mbps or 0.0),
        "segment_popularity": float(segment.popularity),
        "service_priority": float(segment.service_priority),
        "buffer_min_s": min(buffers),
        "buffer_mean_s": statistics.mean(buffers),
        "sinr_mean_db": statistics.mean(sinr_db),
        "sinr_min_db": min(sinr_db),
        "shannon_mean_mbps": statistics.mean(shannon_cell),
        "shannon_min_mbps": min(shannon_cell),
        "mobility_risk_mean": statistics.mean(risks),
        "mobility_risk_max": max(risks),
        "cell_capacity_mbps": cell.maximum_capacity_mbps,
        "broadcast_capacity_mbps": broadcast_capacity,
        "unicast_load_estimate_mbps": unicast_load,
        "congestion_estimate": compute_congestion(unicast_load, cell.maximum_capacity_mbps),
        "prev_congestion": prev_congestion,
        "prev_broadcast_groups": float(prev_broadcast_groups),
    }

    # If this group broadcasts, it joins the streams already on the
    # carrier — hence +1 over what was observed last tick.
    broadcast_sharing_groups = max(prev_broadcast_groups, 0) + 1

    return GroupFeatures(
        values=values,
        n_requesting=n_requesting,
        segment_bitrate_mbps=float(segment.bitrate_mbps or 0.0),
        buffer_min_s=min(buffers),
        shannon_unicast_mean_mbps=statistics.mean(shannon_cell),
        shannon_broadcast_worst_mbps=min(shannon_bc),
        unicast_sharing_users=n_active,
        broadcast_sharing_groups=broadcast_sharing_groups,
    )


def features_to_vector(features: GroupFeatures) -> list[float]:
    return features.vector()
