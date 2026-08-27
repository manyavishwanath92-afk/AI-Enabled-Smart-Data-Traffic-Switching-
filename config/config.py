"""Central simulation configuration.

`SimConfig` is the single source of truth for every tunable parameter in
the Phase 1 simulator. Components must never hardcode a value that has a
field here — they take a `SimConfig` instance (or the specific field) as
an argument and read from it, so behaviour can be changed entirely from
configuration without touching simulation/network/delivery logic.

IMPORTANT SIMPLIFICATION NOTICE: several defaults below are simplified
PoC assumptions carried over from an earlier winning buildathon PoC
(transmit power, noise floor, interference coefficient, path-loss
constants). They are NOT real-world telecom constants or
standards-derived values — they exist so the simulator has *a* consistent,
documented radio model to reason about, and every one of them is meant to
be experimented with by changing this file, not the code that consumes it.

INTERFERENCE MODEL — "noise rise" (interference-over-thermal):

    I_linear = interference_rise_coeff * active_users * noise_linear

Interference is expressed *relative to the noise floor* rather than as
an absolute milliwatt figure, which makes it dimensionally consistent
with the signal and noise terms it is summed against. At the defaults
below this yields SINR spanning roughly 5-41 dB, falling with both
distance and cell load.

An earlier, dimensionally-unsound formula (`coeff * active_users` with
no noise term, carried over verbatim from an older PoC) produced a bare
milliwatt figure roughly eight orders of magnitude larger than a
realistic received signal, which collapsed SINR to ~1e-9 regardless of
distance or load and left the delivery chain unable to move any content
at all. It has been removed rather than kept selectable — this simulator
implements one interference model, and it is this one.

SINR affects service quality only through the delivery chain (Shannon
capacity -> achievable rate -> buffer -> playback stalls). Service
continuity is measured from actual stalls; see kpi.reliability.

Still a simplified PoC model — not a physical multi-cell interference
calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, fields
from typing import Any


@dataclass
class SimConfig:
    # --- Population scale (Guide Step 4) ---
    num_cells: int = 3
    num_ues: int = 100
    num_segments: int = 10

    # --- Geography (synthetic 2D Cartesian plane, km) ---
    cell_area_radius_km: float = 2.0

    # --- Bandwidth (Guide Section 9.1 / 27 — project simulation notes) ---
    cellular_bandwidth_hz: float = 10_000_000.0  # 10 MHz
    broadcast_bandwidth_hz: float = 6_000_000.0  # 6 MHz

    # --- Radio chain (Guide Section 8 / 12.2 / 27 — old-PoC starting values) ---
    tx_power_dbm: float = 46.0
    noise_floor_dbm: float = -104.0
    # Explicitly configurable PoC assumption, NOT a real-world telecom
    # interference constant — see module docstring for the noise-rise
    # formula. interference_linear = interference_rise_coeff *
    # active_users * noise_linear. Change this value to experiment with
    # interference sensitivity — no code changes needed.
    interference_rise_coeff: float = 0.02
    path_loss_const_a: float = 128.1
    path_loss_const_b: float = 37.6

    # --- Resource / capacity model (Guide Section 9, 26 — explicit & configurable,
    #     deliberately NOT reusing the old PoC's fixed 50 Mbps cell capacity /
    #     0.5 Mbps user-rate constants) ---
    cell_capacity_mbps: float = 100.0
    broadcast_cell_capacity_mbps: float = 60.0
    # Normalized per-stream resource COST used by resource accounting and
    # congestion (R_unicast = N * this, R_broadcast = this). Distinct from
    # a segment's actual content bitrate (Segment.bitrate_mbps), which is
    # what the content/buffer chain uses — see content/segment.py.
    segment_resource_mbps: float = 2.0

    # --- Content bitrate range (Phase 2) ---
    # ContentServer draws each segment's bitrate from this range and then
    # derives size_mb = bitrate * duration / 8, so size, duration and
    # bitrate are mutually consistent.
    segment_bitrate_mbps_min: float = 0.5
    segment_bitrate_mbps_max: float = 4.0

    # --- Buffer / transition safety (Guide Section 10, 18.5, 27) ---
    # Seeds each UE's buffer_seconds ONCE at construction time
    # (traffic_sim.simulation.traffic.create_ues). After that every UE's
    # buffer evolves independently tick-by-tick and this value is never
    # re-read for that UE.
    initial_buffer_seconds: float = 5.0
    transition_time_seconds: float = 2.0
    safety_margin_seconds: float = 1.0
    playback_consumption_rate: float = 1.0  # seconds of content consumed per second of playback (real-time)

    # --- Mobility (Guide Section 10.4) ---
    ue_mobility_speed_mps_min: float = 0.0
    ue_mobility_speed_mps_max: float = 5.0

    # --- Timing / reporting (Guide Section 3.2, 27) ---
    simulation_timestep_s: float = 1.0
    simulation_duration_s: float = 300.0
    reporting_interval_s: float = 1.0
    popularity_update_interval_s: float = 30.0

    # --- Reproducibility ---
    random_seed: int = 42

    # --- AI decision layer (traffic_sim.ai) ---
    # A switch must improve utility by more than this to be worth making;
    # stops the agent flip-flopping on near-ties.
    hysteresis_margin: float = 0.05
    # After switching a group's mode, hold it for this long before
    # switching again.
    cooldown_period_s: float = 5.0
    # Below this confidence the agent keeps the current mode and logs that
    # it declined to act.
    min_decision_confidence: float = 0.15
    # Directory holding the trained XGBoost outcome models. If the files
    # are missing the agent falls back to computing outcomes analytically.
    ai_model_path: str = "models"
    # Utility weights (see traffic_sim.ai.utility). None -> DEFAULT_WEIGHTS.
    utility_weights: dict[str, float] | None = None

    # --- PCF policy layer (traffic_sim.policy.pcf) ---
    # Segments at or above this service_priority get a stricter buffer
    # margin before PCF permits any mode change this tick — delay-sensitive
    # content gets a stronger continuity guarantee than the agent's single
    # fixed margin provides (Guide Section 10.5).
    pcf_priority_threshold: int = 1
    # How much larger that margin is, as a multiple of the agent's base
    # (transition_time_seconds + safety_margin_seconds).
    pcf_priority_margin_multiplier: float = 1.5

    def validate(self) -> None:
        """Raise ValueError on any structurally unsound configuration."""
        if self.num_cells <= 0:
            raise ValueError("num_cells must be > 0")
        if self.num_ues <= 0:
            raise ValueError("num_ues must be > 0")
        if self.num_segments <= 0:
            raise ValueError("num_segments must be > 0")
        if self.cellular_bandwidth_hz <= 0 or self.broadcast_bandwidth_hz <= 0:
            raise ValueError("bandwidths must be > 0")
        if self.cell_capacity_mbps <= 0 or self.broadcast_cell_capacity_mbps <= 0:
            raise ValueError("cell capacities must be > 0")
        if self.segment_resource_mbps <= 0:
            raise ValueError("segment_resource_mbps must be > 0")
        if self.interference_rise_coeff < 0:
            raise ValueError("interference_rise_coeff must be >= 0")
        if self.hysteresis_margin < 0:
            raise ValueError("hysteresis_margin must be >= 0")
        if self.cooldown_period_s < 0:
            raise ValueError("cooldown_period_s must be >= 0")
        if not 0.0 <= self.min_decision_confidence <= 1.0:
            raise ValueError("min_decision_confidence must be in [0, 1]")
        if self.pcf_priority_threshold < 0:
            raise ValueError("pcf_priority_threshold must be >= 0")
        if self.pcf_priority_margin_multiplier < 1.0:
            raise ValueError("pcf_priority_margin_multiplier must be >= 1.0")
        if self.segment_bitrate_mbps_min <= 0:
            raise ValueError("segment_bitrate_mbps_min must be > 0")
        if self.segment_bitrate_mbps_max < self.segment_bitrate_mbps_min:
            raise ValueError("segment_bitrate_mbps_max must be >= segment_bitrate_mbps_min")
        if self.initial_buffer_seconds < 0:
            raise ValueError("initial_buffer_seconds must be >= 0")
        if self.transition_time_seconds < 0 or self.safety_margin_seconds < 0:
            raise ValueError("transition_time_seconds/safety_margin_seconds must be >= 0")
        if self.simulation_timestep_s <= 0:
            raise ValueError("simulation_timestep_s must be > 0")
        if self.simulation_duration_s <= 0:
            raise ValueError("simulation_duration_s must be > 0")
        if self.cell_area_radius_km <= 0:
            raise ValueError("cell_area_radius_km must be > 0")
        if self.ue_mobility_speed_mps_min < 0 or self.ue_mobility_speed_mps_max < self.ue_mobility_speed_mps_min:
            raise ValueError("invalid ue_mobility_speed_mps range")

    @classmethod
    def default(cls) -> "SimConfig":
        cfg = cls()
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def with_overrides(self, **overrides: Any) -> "SimConfig":
        """Return a new SimConfig with the given fields replaced (validated)."""
        cfg = replace(self, **overrides)
        cfg.validate()
        return cfg
