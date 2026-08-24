"""Synthetic content server (Guide Section 7.1, 6.4).

Generates ~num_segments synthetic Segment objects with varied size and
duration, and a skewed initial popularity so some segments start
unpopular and some popular. `drift_popularity` periodically reshuffles
which segments are "hot" so popularity changes over time, per the
guide's traffic-model requirement. No real video content is produced or
required.
"""

from __future__ import annotations

import random

from config.config import SimConfig
from traffic_sim.content.segment import Segment


class ContentServer:
    def __init__(self, config: SimConfig, rng: random.Random | None = None):
        self._config = config
        self._rng = rng or random.Random(config.random_seed)
        self._segments: dict[str, Segment] = {}
        self._seconds_since_drift = 0.0
        self._generate_segments()

    def _generate_segments(self) -> None:
        for i in range(self._config.num_segments):
            segment_id = f"seg_{i}"
            size_mb = self._rng.uniform(1.0, 8.0)
            duration_sec = self._rng.uniform(5.0, 30.0)
            priority = self._rng.choice([0, 0, 1, 2])  # skewed toward delay-tolerant
            self._segments[segment_id] = Segment(
                segment_id=segment_id,
                size_mb=size_mb,
                duration_sec=duration_sec,
                popularity=0.0,
                service_priority=priority,
            )
        # Zipf-like initial popularity skew: rank segments, weight ~1/rank.
        ranked_ids = list(self._segments.keys())
        self._rng.shuffle(ranked_ids)
        weights = [1.0 / (rank + 1) for rank in range(len(ranked_ids))]
        total = sum(weights)
        for segment_id, w in zip(ranked_ids, weights):
            self._segments[segment_id].popularity = w / total

    def get_segment(self, segment_id: str) -> Segment:
        return self._segments[segment_id]

    def all_segments(self) -> list[Segment]:
        return list(self._segments.values())

    def hottest_segment_id(self) -> str:
        return max(self._segments.values(), key=lambda s: s.popularity).segment_id

    def drift_popularity(self, dt: float) -> None:
        """Reshuffle which segments are "hot" every popularity_update_interval_s."""
        self._seconds_since_drift += dt
        if self._seconds_since_drift < self._config.popularity_update_interval_s:
            return
        self._seconds_since_drift = 0.0
        ranked_ids = list(self._segments.keys())
        self._rng.shuffle(ranked_ids)
        weights = [1.0 / (rank + 1) for rank in range(len(ranked_ids))]
        total = sum(weights)
        for segment_id, w in zip(ranked_ids, weights):
            self._segments[segment_id].popularity = w / total
