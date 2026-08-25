"""UE (User Equipment): a simulated device (Guide Section 6.3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MobilityState(Enum):
    STATIONARY = "stationary"
    WALKING = "walking"
    VEHICULAR = "vehicular"


class DeliveryMode(Enum):
    NONE = "none"
    UNICAST = "unicast"
    BROADCAST = "broadcast"


class QoEState(Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    STARVED = "starved"


class RadioState(Enum):
    CONNECTED = "connected"
    WEAK = "weak"
    DISCONNECTED = "disconnected"


@dataclass
class UE:
    ue_id: str
    x_km: float
    y_km: float
    cell_id: str | None = None
    requested_segment: str | None = None
    buffer_seconds: float = 0.0
    mobility_state: MobilityState = MobilityState.STATIONARY
    delivery_mode: DeliveryMode = DeliveryMode.NONE
    qoe_state: QoEState = QoEState.GOOD
    radio_state: RadioState = RadioState.CONNECTED
    velocity_mps: float = 0.0
    direction_rad: float = 0.0
    sinr_linear: float | None = None
    mobility_risk: float | None = None
    # Playback continuity accounting (Guide Section 11): a UE stalls on a
    # tick when its buffer is empty, and a rebuffering event is counted on
    # the transition from playing to stalled (not once per stalled tick).
    stall_ticks: int = 0
    playback_ticks: int = 0
    rebuffering_events: int = 0
    is_stalled: bool = False
