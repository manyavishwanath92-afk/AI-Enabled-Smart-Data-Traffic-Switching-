"""Subscription management: which UEs want which segment (Guide Section 7.2, 7.3).

requests = {segment_id: [ue_id, ...]}
Popularity(segment) = requesting_UEs_for_segment / active_UEs

Requests are per-tick, not cumulative: `clear_requests` is called before
each tick's demand is regenerated, so popularity reflects *current*
demand rather than an ever-growing historical count.
"""

from __future__ import annotations

from traffic_sim.content.content_server import ContentServer


class SubscriptionManager:
    def __init__(self, content_server: ContentServer):
        self._content_server = content_server
        self._requests: dict[str, list[str]] = {}

    def clear_requests(self) -> None:
        self._requests = {}

    def record_request(self, segment_id: str, ue_id: str) -> None:
        self._requests.setdefault(segment_id, []).append(ue_id)

    def requesting_ues_for_segment(self, segment_id: str) -> list[str]:
        return list(self._requests.get(segment_id, []))

    def all_requests(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._requests.items()}

    def compute_popularity(self, active_ue_count: int) -> dict[str, float]:
        """Popularity(segment) = requesting_UEs_for_segment / active_UEs.

        Writes the computed value back onto each Segment in the
        ContentServer (overwriting the content server's synthetic drifted
        value with the actually-observed popularity for this tick).
        """
        result: dict[str, float] = {}
        for segment in self._content_server.all_segments():
            requesting = len(self._requests.get(segment.segment_id, []))
            popularity = (requesting / active_ue_count) if active_ue_count > 0 else 0.0
            segment.popularity = popularity
            result[segment.segment_id] = popularity
        return result
