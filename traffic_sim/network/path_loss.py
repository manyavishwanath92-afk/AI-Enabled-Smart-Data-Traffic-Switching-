"""Distance -> path loss (Guide Section 8.1).

PL(dB) = a + b * log10(d_km)

Simplified urban-macrocell relation carried over from the old PoC. This
is a PoC simplification, NOT a full 3GPP/ATSC physical-layer model.
"""

from __future__ import annotations

import math

MIN_DISTANCE_KM = 1e-6  # guards log10(0); no UE is ever exactly at the tower


def path_loss_db(distance_km: float, a: float, b: float) -> float:
    """PL(dB) = a + b * log10(d_km)."""
    d = max(distance_km, MIN_DISTANCE_KM)
    return a + b * math.log10(d)
