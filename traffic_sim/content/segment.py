"""Segment: synthetic content-segment metadata (Guide Section 7.1).

`bitrate_mbps` is the rate needed to sustain real-time playback and is
what the content/buffer chain uses (see network.throughput). It is
distinct from config.segment_resource_mbps, which is the normalized
per-stream resource COST used by resource accounting and congestion.

ContentServer generates segments so that size_mb, duration_sec and
bitrate_mbps are mutually consistent (size = bitrate * duration / 8). If
a Segment is constructed directly without an explicit bitrate — as some
tests do — it is derived from size and duration instead, so the identity
always holds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    segment_id: str
    size_mb: float
    duration_sec: float
    bitrate_mbps: float | None = None
    popularity: float = 0.0
    # 0 = delay-tolerant .. higher = more delay-sensitive. Numeric encoding
    # only; not consumed by any Phase 1 logic (reserved for the utility
    # model), but must exist per the guide's requirement that service
    # priority is part of the segment/UE context.
    service_priority: int = 0

    def __post_init__(self) -> None:
        if self.bitrate_mbps is None:
            self.bitrate_mbps = (
                (self.size_mb * 8.0) / self.duration_sec if self.duration_sec > 0 else 0.0
            )
