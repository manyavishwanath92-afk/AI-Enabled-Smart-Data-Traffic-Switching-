"""KPIEngine: aggregates all Phase 1 KPIs into one per-tick record (Guide Step 19, Section 11).

Fields that only become meaningful once an AI agent exists
(`switch_count`, `ai_inference_latency_ms`) are present in every record
now, always None/0 in Phase 1, so the schema is stable going into
Phase 2 — see traffic_sim/ai/__init__.py.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from config.config import SimConfig
from traffic_sim.delivery.result import DeliveryResult
from traffic_sim.kpi.bandwidth import resource_savings, spectrum_efficiency
from traffic_sim.kpi.qoe import compute_qoe
from traffic_sim.kpi.reliability import (
    buffer_starvation_rate,
    observe_playback,
    service_continuity,
    successful_delivery_ratio,
)
from traffic_sim.network.sinr import sinr_to_db
from traffic_sim.simulation.buffer import group_safety_buffer
from traffic_sim.simulation.user import UE


@dataclass
class KPIRecord:
    tick: int
    sim_time_s: float
    mode_breakdown: str  # e.g. "unicast=2,broadcast=1" — human-readable summary of this tick's groups
    num_groups: int
    num_ues_delivered: int
    mean_congestion: float
    total_required_resource_mbps: float
    resource_baseline_always_unicast_mbps: float
    resource_savings_vs_baseline: float
    mean_buffer_seconds: float
    min_group_buffer_seconds: float
    buffer_starvation_rate: float
    # SC = 1 - (playback interruption time / total playback time). The
    # single definition of service continuity, measured from actual stalls.
    service_continuity: float
    rebuffering_events: int
    # Phase 2 radio-chain visibility: what SINR the UEs actually saw and
    # what delivery rate that translated into after resource sharing.
    mean_sinr_db: float
    mean_achievable_mbps: float
    successful_delivery_ratio: float
    mean_spectrum_efficiency: float
    mean_qoe: float
    # Populated when the delivery-mode selector reports decision telemetry
    # (the AI agent does; fixed-rule selectors leave these as None).
    switch_count: int | None = None
    ai_inference_latency_ms: float | None = None


class KPIEngine:
    def __init__(self, config: SimConfig):
        self._config = config
        self._records: list[KPIRecord] = []

    def record_tick(
        self,
        tick: int,
        sim_time_s: float,
        ues: list[UE],
        delivery_results: list[DeliveryResult],
        dt: float,
        decision_stats: object | None = None,
    ) -> KPIRecord:
        cfg = self._config

        mode_counts: dict[str, int] = {}
        for r in delivery_results:
            mode_counts[r.mode] = mode_counts.get(r.mode, 0) + 1
        mode_breakdown = ",".join(f"{m}={c}" for m, c in sorted(mode_counts.items())) or "none"

        total_requesting = sum(r.n_requesting for r in delivery_results)
        mean_congestion = (
            sum(r.congestion * r.n_requesting for r in delivery_results) / total_requesting
            if total_requesting > 0
            else 0.0
        )

        total_required = sum(r.required_resource_mbps for r in delivery_results)
        baseline_always_unicast = sum(r.n_requesting * cfg.segment_resource_mbps for r in delivery_results)
        savings = resource_savings(total_required, baseline_always_unicast)

        num_ues_delivered = sum(len(r.delivered_ue_ids) for r in delivery_results)
        successful_ratio = successful_delivery_ratio(num_ues_delivered, total_requesting)

        mean_buffer = statistics.mean(ue.buffer_seconds for ue in ues) if ues else 0.0
        min_group_buffer = group_safety_buffer(ues)
        starvation = buffer_starvation_rate(ues)

        # Radio quality, reported on its own — it influences service
        # continuity only through the delivery chain.
        sinr_observations = [ue.sinr_linear for ue in ues if ue.sinr_linear is not None]
        mean_sinr_db = (
            statistics.mean(sinr_to_db(s) for s in sinr_observations if s > 0)
            if any(s > 0 for s in sinr_observations)
            else 0.0
        )

        # Advance playback counters once per tick, then report continuity.
        observe_playback(ues)
        continuity = service_continuity(ues)
        total_rebuffering = sum(ue.rebuffering_events for ue in ues)

        achievable_values = [
            v for r in delivery_results for v in r.achievable_mbps_by_ue.values()
        ]
        mean_achievable = statistics.mean(achievable_values) if achievable_values else 0.0

        se_values = []
        for r in delivery_results:
            bw_hz = cfg.broadcast_bandwidth_hz if r.mode == "broadcast" else cfg.cellular_bandwidth_hz
            delivered_bits = r.required_resource_mbps * 1_000_000.0 * dt
            se_values.append(spectrum_efficiency(delivered_bits, bw_hz, dt))
        mean_se = statistics.mean(se_values) if se_values else 0.0

        mean_qoe = statistics.mean(compute_qoe(ue) for ue in ues) if ues else 0.0

        record = KPIRecord(
            tick=tick,
            sim_time_s=sim_time_s,
            mode_breakdown=mode_breakdown,
            num_groups=len(delivery_results),
            num_ues_delivered=num_ues_delivered,
            mean_congestion=mean_congestion,
            total_required_resource_mbps=total_required,
            resource_baseline_always_unicast_mbps=baseline_always_unicast,
            resource_savings_vs_baseline=savings,
            mean_buffer_seconds=mean_buffer,
            min_group_buffer_seconds=min_group_buffer,
            buffer_starvation_rate=starvation,
            service_continuity=continuity,
            rebuffering_events=total_rebuffering,
            mean_sinr_db=mean_sinr_db,
            mean_achievable_mbps=mean_achievable,
            successful_delivery_ratio=successful_ratio,
            mean_spectrum_efficiency=mean_se,
            mean_qoe=mean_qoe,
            switch_count=getattr(decision_stats, "switch_count", None),
            ai_inference_latency_ms=getattr(decision_stats, "mean_inference_latency_ms", None),
        )
        self._records.append(record)
        return record

    def to_records(self) -> list[dict]:
        return [asdict(r) for r in self._records]

    def export_csv(self, path: str | Path) -> None:
        records = self.to_records()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            path.write_text("")
            return
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    def summary(self) -> dict:
        if not self._records:
            return {}
        fields_to_average = [
            "mean_congestion",
            "resource_savings_vs_baseline",
            "mean_buffer_seconds",
            "min_group_buffer_seconds",
            "buffer_starvation_rate",
            "service_continuity",
            "mean_sinr_db",
            "mean_achievable_mbps",
            "successful_delivery_ratio",
            "mean_spectrum_efficiency",
            "mean_qoe",
        ]
        result = {}
        for field_name in fields_to_average:
            values = [getattr(r, field_name) for r in self._records]
            result[field_name] = statistics.mean(values)
        result["total_required_resource_mbps"] = sum(r.total_required_resource_mbps for r in self._records)
        result["total_resource_baseline_always_unicast_mbps"] = sum(
            r.resource_baseline_always_unicast_mbps for r in self._records
        )
        result["total_rebuffering_events"] = (
            self._records[-1].rebuffering_events if self._records else 0
        )
        switches = [r.switch_count for r in self._records if r.switch_count is not None]
        latencies = [
            r.ai_inference_latency_ms for r in self._records if r.ai_inference_latency_ms is not None
        ]
        result["total_switches"] = sum(switches) if switches else 0
        result["mean_inference_latency_ms"] = (
            statistics.mean(latencies) if latencies else 0.0
        )
        result["num_ticks"] = len(self._records)
        return result
