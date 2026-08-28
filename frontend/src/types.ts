export type RunStatus = "pending" | "running" | "paused" | "completed" | "stopped" | "failed";
export type Mode = "unicast" | "broadcast";

export interface SimConfig {
  num_cells?: number;
  num_ues?: number;
  num_segments?: number;
  cellular_bandwidth_hz?: number;
  broadcast_bandwidth_hz?: number;
  cell_capacity_mbps?: number;
  broadcast_cell_capacity_mbps?: number;
  hysteresis_margin?: number;
  cooldown_period_s?: number;
  min_decision_confidence?: number;
  pcf_priority_threshold?: number;
  pcf_priority_margin_multiplier?: number;
  [key: string]: unknown;
}

export interface SimulationStatus {
  simulation_id: string;
  status: RunStatus;
  current_tick: number;
  total_ticks: number;
  scenario: string;
  mode_selector: string;
  tick_interval_s: number;
  seed?: number;
  error?: string | null;
}

export interface Cell {
  cell_id: string;
  center_x_km: number;
  center_y_km: number;
  maximum_capacity_mbps: number;
  broadcast_capacity_mbps: number;
  current_load_mbps: number;
  congestion: number;
  connected_ue_ids?: string[];
  num_connected_ues: number;
}

export interface UE {
  ue_id: string;
  x_km: number;
  y_km: number;
  cell_id: string;
  requested_segment: string | null;
  buffer_seconds: number;
  mobility_state: string;
  delivery_mode: "none" | Mode;
  qoe_state: "good" | "degraded" | "starved" | string;
  radio_state: "connected" | "weak" | "disconnected" | string;
  velocity_mps: number;
  direction_rad: number;
  sinr_linear: number | null;
  mobility_risk: number;
  stall_ticks: number;
  playback_ticks: number;
  rebuffering_events: number;
  is_stalled?: boolean;
}

export interface KPIRecord {
  tick: number;
  sim_time_s: number;
  mode_breakdown: string;
  num_groups: number;
  num_ues_delivered: number;
  mean_congestion: number;
  total_required_resource_mbps: number;
  resource_baseline_always_unicast_mbps: number;
  resource_savings_vs_baseline: number;
  mean_buffer_seconds: number;
  min_group_buffer_seconds: number;
  buffer_starvation_rate: number;
  service_continuity: number;
  rebuffering_events: number;
  mean_sinr_db: number;
  mean_achievable_mbps: number;
  successful_delivery_ratio: number;
  mean_spectrum_efficiency: number;
  mean_qoe: number;
  switch_count: number | null;
  ai_inference_latency_ms: number | null;
}

export interface Decision {
  cell_id: string;
  group_id: string;
  selected_mode: Mode;
  previous_mode: Mode | null;
  switched: boolean;
  confidence: number;
  explanation: string;
  bearer: "cellular" | "broadcast" | string;
  policy_compliant: boolean;
}

export interface DecisionHistoryRow extends Decision {
  tick: number;
  sim_time_s: number;
  utility_unicast?: number;
  utility_broadcast?: number;
  broadcast_benefit?: number;
  decision_reason?: string;
  mbos_bearer?: string;
  pcf_result?: string;
  inference_latency_ms?: number;
}

export interface DeliveryGroup {
  mode: Mode;
  segment_id: string;
  cell_id: string;
  n_requesting: number;
  required_resource_mbps: number;
  congestion: number;
  sinr_by_ue: Record<string, number>;
  achievable_mbps_by_ue: Record<string, number>;
  content_received_seconds_by_ue: Record<string, number>;
  delivered_ue_ids: string[];
}

export interface TickMessage {
  type: "tick";
  simulation_id: string;
  tick: number;
  kpi: KPIRecord;
  cells: Cell[];
  decisions: Decision[];
  groups: DeliveryGroup[];
}

export interface StatusMessage extends SimulationStatus {
  type: "status";
}

export interface ComparisonRow {
  policy: string;
  summary: Record<string, number | string | null>;
}

export interface UeResponse {
  total: number;
  limit: number;
  offset: number;
  items: UE[];
}

export interface CreateSimulationRequest {
  scenario: string;
  mode_selector: string;
  ticks: number;
  seed: number;
  tick_interval_s: number;
  config_overrides?: Record<string, number | string | boolean>;
}
