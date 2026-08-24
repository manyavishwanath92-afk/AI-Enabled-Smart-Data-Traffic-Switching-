from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import DeliveryMode, MobilityState, UE


def test_cell_congestion_normal():
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=100.0, current_load_mbps=50.0)
    assert cell.congestion() == 0.5


def test_cell_congestion_clamped_at_overload():
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=50.0, current_load_mbps=75.0)
    assert cell.congestion() == 1.0
    # current_load itself is NOT clamped, for diagnostics
    assert cell.current_load_mbps == 75.0


def test_cell_connect_disconnect():
    cell = Cell(cell_id="c0", center_x_km=0, center_y_km=0, maximum_capacity_mbps=100.0)
    cell.connect_ue("ue_1")
    assert "ue_1" in cell.connected_ue_ids
    cell.disconnect_ue("ue_1")
    assert "ue_1" not in cell.connected_ue_ids


def test_ue_defaults():
    ue = UE(ue_id="ue_1", x_km=0.0, y_km=0.0)
    assert ue.mobility_state == MobilityState.STATIONARY
    assert ue.delivery_mode == DeliveryMode.NONE
    assert ue.buffer_seconds == 0.0
