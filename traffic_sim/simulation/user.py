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
