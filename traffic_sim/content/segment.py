"""Segment: synthetic content-segment metadata (Guide Section 7.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    segment_id: str
    size_mb: float
    duration_sec: float
    popularity: float = 0.0
    # 0 = delay-tolerant .. higher = more delay-sensitive. Numeric encoding
    # only; not consumed by any Phase 1 logic (reserved for Phase 2 utility
    # model weighting), but must exist per the guide's requirement that
    # service priority is part of the segment/UE context.
    service_priority: int = 0
