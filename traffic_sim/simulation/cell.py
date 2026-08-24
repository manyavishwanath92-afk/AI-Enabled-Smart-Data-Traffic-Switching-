"""Cell: a modeled coverage/resource area (Guide Section 6.2).

Coordinates are a synthetic 2D Cartesian plane in kilometres, not real
geography — this keeps distance-based radio math (path loss) simple
without implying a real deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Cell:
    cell_id: str
    center_x_km: float
    center_y_km: float
    maximum_capacity_mbps: float
    broadcast_capacity_mbps: float | None = None
    current_load_mbps: float = 0.0
    connected_ue_ids: set[str] = field(default_factory=set)

    def congestion(self) -> float:
        """congestion = current_load / maximum_capacity, clamped to [0, 1].

        Convention (documented once, used everywhere): the reported KPI is
        clamped to [0, 1] even under overload. `current_load_mbps` itself
        is left unclamped so diagnostics can still see true demand above
        capacity.
        """
        if self.maximum_capacity_mbps <= 0:
            return 1.0
        return min(1.0, max(0.0, self.current_load_mbps / self.maximum_capacity_mbps))

    def reset_load(self) -> None:
        self.current_load_mbps = 0.0

    def add_load(self, mbps: float) -> None:
        self.current_load_mbps += mbps

    def connect_ue(self, ue_id: str) -> None:
        self.connected_ue_ids.add(ue_id)

    def disconnect_ue(self, ue_id: str) -> None:
        self.connected_ue_ids.discard(ue_id)
