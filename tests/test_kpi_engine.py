from config.config import SimConfig
from traffic_sim.delivery.result import DeliveryResult
from traffic_sim.kpi.metrics import KPIEngine
from traffic_sim.kpi.qoe import compute_qoe
from traffic_sim.network.sinr import sinr_threshold_linear
from traffic_sim.simulation.user import UE


def test_kpi_record_tick_produces_placeholders_for_phase2_fields():
    cfg = SimConfig.default()
    engine = KPIEngine(cfg)
    ues = [UE(ue_id="ue_1", x_km=0, y_km=0, buffer_seconds=5.0)]
    result = DeliveryResult(
        mode="unicast",
        segment_id="s0",
        cell_id="c0",
        n_requesting=1,
        required_resource_mbps=2.0,
        congestion=0.1,
        delivered_ue_ids=["ue_1"],
    )
    record = engine.record_tick(tick=1, sim_time_s=1.0, ues=ues, delivery_results=[result], dt=1.0)

    assert record.switch_count is None
    assert record.ai_inference_latency_ms is None
    assert record.num_ues_delivered == 1


def test_resource_savings_positive_when_broadcast_cheaper():
    cfg = SimConfig.default().with_overrides(segment_resource_mbps=2.0)
    engine = KPIEngine(cfg)
    ues = [UE(ue_id=f"ue_{i}", x_km=0, y_km=0, buffer_seconds=5.0) for i in range(10)]
    broadcast_result = DeliveryResult(
        mode="broadcast",
        segment_id="s0",
        cell_id="c0",
        n_requesting=10,
        required_resource_mbps=2.0,  # flat, independent of group size
        congestion=0.05,
        delivered_ue_ids=[u.ue_id for u in ues],
    )
    record = engine.record_tick(tick=1, sim_time_s=1.0, ues=ues, delivery_results=[broadcast_result], dt=1.0)

    # baseline (always-unicast) would have been 10 * 2.0 = 20 Mbps
    assert record.resource_baseline_always_unicast_mbps == 20.0
    assert record.resource_savings_vs_baseline > 0.0


def test_compute_qoe_monotonic_in_buffer():
    cfg = SimConfig.default()
    threshold = sinr_threshold_linear(cfg.sinr_threshold_db)
    low_buffer_ue = UE(ue_id="a", x_km=0, y_km=0, buffer_seconds=1.0, sinr_linear=threshold)
    high_buffer_ue = UE(ue_id="b", x_km=0, y_km=0, buffer_seconds=30.0, sinr_linear=threshold)

    low_qoe = compute_qoe(low_buffer_ue, threshold)
    high_qoe = compute_qoe(high_buffer_ue, threshold)

    assert high_qoe >= low_qoe


def test_compute_qoe_starved_state_on_zero_buffer():
    from traffic_sim.simulation.user import QoEState

    cfg = SimConfig.default()
    threshold = sinr_threshold_linear(cfg.sinr_threshold_db)
    ue = UE(ue_id="a", x_km=0, y_km=0, buffer_seconds=0.0)
    compute_qoe(ue, threshold)
    assert ue.qoe_state == QoEState.STARVED


def test_playback_continuity_and_radio_kpis_are_recorded():
    cfg = SimConfig.default()
    engine = KPIEngine(cfg)
    threshold = sinr_threshold_linear(cfg.sinr_threshold_db)
    healthy = UE(ue_id="ok", x_km=0, y_km=0, buffer_seconds=5.0, sinr_linear=threshold * 4)
    starved = UE(ue_id="dry", x_km=0, y_km=0, buffer_seconds=0.0, sinr_linear=threshold * 4)
    result = DeliveryResult(
        mode="unicast", segment_id="s0", cell_id="c0", n_requesting=2,
        required_resource_mbps=4.0, congestion=0.2,
        achievable_mbps_by_ue={"ok": 2.0, "dry": 0.5},
        delivered_ue_ids=["ok", "dry"],
    )
    record = engine.record_tick(
        tick=1, sim_time_s=1.0, ues=[healthy, starved], delivery_results=[result], dt=1.0
    )

    # One of two UEs stalled this tick.
    assert record.playback_continuity == 0.5
    assert record.rebuffering_events == 1
    assert record.mean_achievable_mbps == 1.25
    assert record.mean_sinr_db > 0


def test_rebuffering_counts_one_event_per_outage_not_per_tick():
    cfg = SimConfig.default()
    engine = KPIEngine(cfg)
    starved = UE(ue_id="dry", x_km=0, y_km=0, buffer_seconds=0.0)
    result = DeliveryResult(
        mode="unicast", segment_id="s0", cell_id="c0", n_requesting=1,
        required_resource_mbps=2.0, congestion=0.1, delivered_ue_ids=["dry"],
    )
    for tick in range(1, 4):  # stalled for three consecutive ticks
        record = engine.record_tick(
            tick=tick, sim_time_s=float(tick), ues=[starved], delivery_results=[result], dt=1.0
        )

    assert starved.stall_ticks == 3
    assert record.rebuffering_events == 1  # one outage, not three
    assert record.playback_continuity == 0.0


def test_export_csv_writes_header_and_row(tmp_path):
    cfg = SimConfig.default()
    engine = KPIEngine(cfg)
    ues = [UE(ue_id="ue_1", x_km=0, y_km=0, buffer_seconds=5.0)]
    result = DeliveryResult(
        mode="unicast", segment_id="s0", cell_id="c0", n_requesting=1,
        required_resource_mbps=2.0, congestion=0.1, delivered_ue_ids=["ue_1"],
    )
    engine.record_tick(tick=1, sim_time_s=1.0, ues=ues, delivery_results=[result], dt=1.0)

    out_path = tmp_path / "run.csv"
    engine.export_csv(out_path)
    content = out_path.read_text()
    assert "tick" in content.splitlines()[0]
    assert len(content.splitlines()) == 2  # header + 1 row
