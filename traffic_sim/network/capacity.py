"""SINR -> Shannon capacity (Guide Section 9.3).

C = B * log2(1 + SINR)

Theoretical channel-capacity upper bound under this simplified model —
not a claim about a real 5G scheduler or ATSC PHY throughput.
"""

from __future__ import annotations

import math


def shannon_capacity_bps(bandwidth_hz: float, sinr_linear: float) -> float:
    """C = B * log2(1 + SINR), B in Hz, SINR linear."""
    return bandwidth_hz * math.log2(1 + sinr_linear)
