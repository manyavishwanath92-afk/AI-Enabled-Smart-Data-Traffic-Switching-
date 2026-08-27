# Frontend / API Handoff — AI-Enabled Smart Data Traffic Switching

**Audience:** Manus (frontend/dashboard implementer). **Status:** this document is a **specification**, not a running system. Nothing in "Part 4 — REST API Contract" is implemented yet; it describes the contract a future API layer should expose over the existing, tested backend described in Parts 1–3. No frontend code, and no backend code beyond what earlier phases already validated (Phase 2A radio/delivery, the intelligence layer, MBOS/PCF), exists as a result of this document.

Every field named below is cited to a real file/function/class in the repository. Nothing here is invented. Where something the dashboard would want isn't currently persisted anywhere queryable, that's called out explicitly as a gap rather than silently assumed.

---

## Part 1 — Project Context, in Simple Terms

**The problem.** When many people want the same video/content at the same time in the same area (a popular live stream, breaking news), sending each of them their own individual copy (**unicast**) wastes network capacity — the same data goes out N times. Sending everyone one shared copy (**broadcast**) is efficient when a lot of people want the same thing, but wasteful when only a few do, and it doesn't adapt to who's *actually* watching, how good their signal is, or whether they're about to move out of range. The right choice depends on live conditions, and it's an AI's job to decide it segment by segment, not a fixed rule.

**What the simulator represents.** This project doesn't touch a real telecom network. It's a self-contained, from-scratch simulation of one: a synthetic population of devices (**UEs** — User Equipment) moving around, connected to a handful of cells, requesting synthetic video content, with a simplified but internally-consistent radio/physics model (signal strength, interference, capacity) that genuinely limits how fast content can be delivered. Every number in the dashboard traces back to this simulation, not to a live network.

**Cellular unicast vs. ATSC 3.0 broadcast.** "Cellular unicast" is the normal path — each device gets its own private data connection through the cell tower, and if 50 devices want the same segment, the tower does 50 separate transmissions. "ATSC 3.0 broadcast" (a real next-gen broadcast TV standard, used here only as the conceptual reference for the broadcast path — no real ATSC hardware/protocol exists in this project) is one shared transmission on a separate broadcast carrier that any number of devices can tune into at once, for the cost of one stream. The tradeoff the AI has to reason about: broadcast is nearly free per extra listener but is a fixed-size shared pipe (a few streams can use it comfortably; too many, and everyone gets a thin, unusable slice).

**The Phase 2A radio/delivery chain.** This is the physics layer that makes the tradeoff real rather than assumed. For every device: its distance from the cell tower determines *path loss* → path loss subtracted from transmit power gives *received signal power* → received power compared against interference and noise gives *SINR* (signal-to-interference-plus-noise ratio, a standard radio-quality number) → SINR run through the Shannon–Hartley formula gives a *theoretical maximum data rate* for that radio link → that rate, shared fairly among however many devices/streams are actually using the same channel at once, gives the *achievable delivery rate* → that rate, compared against how fast the video needs to arrive to play smoothly (its bitrate), determines how much of the device's playback buffer fills or drains each tick → a buffer that hits zero is a **stall** (rebuffering). This chain is what makes broadcast genuinely better in some conditions and genuinely worse in others — the simulator isn't scripted to prefer one.

**XGBoost's role.** The chain above is exact and computable, but only *inside the simulator* — a real network can't try both delivery modes and keep whichever worked better. So two XGBoost models (a form of machine learning that fits many small decision-rule trees together, good at learning numeric patterns from data) are trained on the simulator's own generated data to *predict* "if this group went unicast, what delivery rate would it get?" and "...if broadcast?" — from live telemetry alone, the way a real system would have to. XGBoost predicts outcomes; it does not decide the mode itself.

**The utility function.** Predicting outcomes isn't the same as deciding which one is *better* — that requires weighing several things at once (does it sustain playback? how much network capacity does it save? is the buffer safe? is this an unnecessary switch?). The utility function turns each predicted outcome into one explainable score, built from named, weighted terms, so any decision can be traced back to *why*.

**The cell-level AI controller.** Early on, letting every content group in a cell decide independently caused a real bug: groups would all pile onto broadcast at once (making it look crowded), all leave at once, then all pile back — a churn loop. The fix was to decide once per cell instead: rank every candidate group by how much it would gain from broadcast, then admit them to the shared broadcast carrier highest-benefit-first, only as long as the carrier can still sustain everyone already admitted. Groups that don't fit stay on unicast.

**MBOS.** After the AI (with cell-level coordination) settles on a mode, MBOS is the orchestration layer that turns that into a concrete delivery instruction: which physical bearer (`cellular` or `broadcast`), a stable session id for that group, and whether this is a **handover** (a bearer change from last tick). It doesn't re-decide anything the AI already decided.

**PCF.** The policy layer, running immediately after MBOS and before the mode is considered truly final. It independently re-checks two things the AI's own logic already tries to get right, as a genuine second opinion: (1) that delay-sensitive content gets a *larger* safety margin before any switch is allowed than ordinary content does, and (2) that the AI's decision confidence really does clear the configured minimum. Either check failing reverts the mode back to whatever it already was.

**Closed-loop feedback.** After delivery actually happens, the resulting congestion and broadcast-carrier crowding are fed back in as inputs to *next* tick's decision — the system observes the consequences of its own actions and adapts, rather than making one-shot, context-free decisions.

---

## Part 2 — The Exact End-to-End Pipeline

```
network state (Simulator.step(), traffic_sim/simulation/simulator.py)
  │
  ▼
feature extraction — traffic_sim/ai/feature_engineering.py :: extract_features()
  │   19 named features per (cell, segment) group: demand, buffers, SINR,
  │   mobility risk, capacity, prior congestion/contention — see Part 3.3
  ▼
two XGBoost predictions (U, B) — traffic_sim/ai/predict.py :: OutcomePredictor.predict()
  │   predicts achievable Mbps for unicast and for broadcast separately;
  │   falls back to the exact-physics analytical evaluator
  │   (traffic_sim/ai/outcomes.py :: evaluate_modes()) if no trained model
  │   is present — PredictionResult.source records which path ran
  ▼
utility scores — traffic_sim/ai/utility.py :: compute_utility()
  │   one score per mode, from named/weighted terms (quality, resource
  │   saving, buffer safety, popularity benefit, mobility risk, switch cost)
  ▼
cell-level broadcast-benefit ranking/admission — traffic_sim/ai/cell_controller.py
  │   :: rank_by_broadcast_benefit(), admit_to_broadcast()
  │   broadcast_benefit = max(0, U_broadcast − U_unicast) per group, ranked
  │   and admitted to the shared broadcast carrier while it still fits
  ▼
AI agent / safety logic — traffic_sim/ai/agent.py :: AIAgent.decide()
  │   buffer safety → confidence floor → hysteresis → cooldown →
  │   stall-escape override; produces the agent's own proposed mode
  ▼
MBOS — traffic_sim/mbos/controller.py :: MBOSController.coordinate() / .confirm()
  │   assigns session_id + target_bearer, flags handover, guarantees a
  │   coherent (never-invalid) instruction
  ▼
PCF — traffic_sim/policy/pcf.py :: PCFPolicy.enforce()
  │   priority-scaled buffer margin + independent confidence-floor check;
  │   may revert to the incumbent mode
  ▼
final U/B decision — traffic_sim/ai/agent.py :: AIDecision (mode, confidence,
  │   reason, session_id, target_bearer, handover, pcf_compliant, ...)
  ▼
delivery — traffic_sim/delivery/unicast.py & broadcast.py :: *.deliver()
  │   invoked by Simulator.step() pass 2 with the coordinated mode
  ▼
SINR / capacity / rate / buffer / playback —
  │   traffic_sim/network/{radio,throughput}.py, traffic_sim/simulation/buffer.py
  │   per-UE SINR → Shannon capacity → achievable rate → buffer fill/drain
  │   → stall/no-stall, mutating each UE and the DeliveryResult for the group
  ▼
KPI calculation — traffic_sim/kpi/metrics.py :: KPIEngine.record_tick()
  │   aggregates congestion, resource savings, service continuity, QoE,
  │   switch count, inference latency, etc. into one KPIRecord for the tick
  ▼
feedback — traffic_sim/ai/agent.py :: AIAgent.observe()
  │   records the congestion and broadcast-carrier occupancy actually
  │   observed, for next tick's feature extraction
  ▼
next decision (loop back to the top for the next tick)
```

---

## Part 3 — Dashboard Field Catalogue

Every field: **name**, **meaning**, **unit**, **source**, **granularity**, **backend-computed or frontend-display-only**. "Backend-computed" means the number already exists as-is in a Python object today; the frontend must render it, never recompute it.

### 3.1 Network Overview

| Field | Meaning | Unit | Source | Granularity | Backend/Frontend |
|---|---|---|---|---|---|
| `tick` | Which simulation step this is | integer | `KPIRecord.tick` (`kpi/metrics.py`) | per-tick | Backend |
| `sim_time_s` | Simulated clock time | seconds | `KPIRecord.sim_time_s` | per-tick | Backend |
| `num_groups` | How many (cell, segment) demand groups existed this tick | count | `KPIRecord.num_groups` | per-tick | Backend |
| `num_ues_delivered` | UEs that received content this tick | count | `KPIRecord.num_ues_delivered` | per-tick | Backend |
| `mode_breakdown` | e.g. `"unicast=2,broadcast=1"` — group counts by mode | string | `KPIRecord.mode_breakdown` | per-tick | Backend |
| `num_cells` | Cells in the simulation | count | `SimConfig.num_cells` (`config/config.py`) | config (constant per run) | Backend |
| `num_ues` | Total simulated devices | count | `SimConfig.num_ues` | config | Backend |
| `num_segments` | Content catalogue size | count | `SimConfig.num_segments` | config | Backend |
| `scenario` | Traffic pattern: `"uniform"` or `"flash_crowd"` | string | `Simulator.scenario` (`simulation/simulator.py`) | per-run | Backend |
| `mode_selector` | Which policy is driving this run (`ai`, `always-unicast`, ...) | string | CLI/API run parameter, see `traffic_sim/main.py::build_simulator` | per-run | Backend |
| `mean_congestion` | How loaded cells are, averaged | 0–1 | `KPIRecord.mean_congestion` / `KPIEngine.summary()["mean_congestion"]` | per-tick or run-average | Backend |

### 3.2 Cell / UE State

**Per cell** (`traffic_sim/simulation/cell.py :: Cell`):

| Field | Meaning | Unit | Granularity | Backend/Frontend |
|---|---|---|---|---|
| `cell_id` | Cell identifier | string | per-cell | Backend |
| `center_x_km`, `center_y_km` | Synthetic 2D position (not real geography) | km | per-cell | Backend |
| `maximum_capacity_mbps` | Cellular (unicast) resource pool | Mbps | per-cell (config-derived) | Backend |
| `broadcast_capacity_mbps` | Broadcast carrier pool | Mbps | per-cell | Backend |
| `current_load_mbps` | Cellular resource currently in use | Mbps | per-cell, per-tick | Backend |
| `congestion` | `Cell.congestion()` = load ÷ capacity, clamped [0,1] | 0–1 | per-cell, per-tick | Backend (method, not a stored field — call it) |
| `connected_ue_count` | `len(Cell.connected_ue_ids)` | count | per-cell, per-tick | Backend |

**Per UE** (`traffic_sim/simulation/user.py :: UE`):

| Field | Meaning | Unit | Granularity | Backend/Frontend |
|---|---|---|---|---|
| `ue_id` | Device identifier | string | per-UE | Backend |
| `x_km`, `y_km` | Position | km | per-UE, per-tick | Backend |
| `cell_id` | Serving cell | string | per-UE, per-tick | Backend |
| `requested_segment` | Content this UE wants right now | string/null | per-UE, per-tick | Backend |
| `buffer_seconds` | Playable content already downloaded | seconds | per-UE, per-tick | Backend |
| `mobility_state` | `stationary` / `walking` / `vehicular` | enum | per-UE, per-tick | Backend |
| `delivery_mode` | Current mode: `none`/`unicast`/`broadcast` | enum | per-UE, per-tick | Backend |
| `qoe_state` | `good` / `degraded` / `starved` | enum | per-UE, per-tick | Backend |
| `radio_state` | `connected` / `weak` / `disconnected` | enum | per-UE, per-tick | Backend |
| `velocity_mps`, `direction_rad` | Movement speed/heading | m/s, radians | per-UE, per-tick | Backend |
| `sinr_linear` | Current SINR (linear, not dB — see `mean_sinr_db` in 3.5 for the dB version) | ratio | per-UE, per-tick | Backend |
| `mobility_risk` | 0 (stationary) to ~1 (likely to leave coverage soon) | 0–1 | per-UE, per-tick | Backend |
| `stall_ticks`, `playback_ticks` | Cumulative stalled / total observed ticks | count | per-UE, cumulative | Backend |
| `rebuffering_events` | Count of stall *episodes* (not stalled ticks — one event per outage) | count | per-UE, cumulative | Backend |
| `is_stalled` | Currently stalled this tick | bool | per-UE, per-tick | Backend |

### 3.3 Unicast vs. Broadcast Usage

| Field | Meaning | Unit | Source | Granularity | Backend/Frontend |
|---|---|---|---|---|---|
| `mode_breakdown` | Group counts by mode this tick | string | `KPIRecord.mode_breakdown` | per-tick | Backend |
| `total_required_resource_mbps` | Total resource cost this tick under the active policy | Mbps | `KPIRecord.total_required_resource_mbps` | per-tick | Backend |
| `resource_baseline_always_unicast_mbps` | What the same tick would have cost if every group had gone unicast | Mbps | `KPIRecord.resource_baseline_always_unicast_mbps` | per-tick | Backend |
| `resource_savings_vs_baseline` | `1 − required/baseline` | 0–1 | `KPIRecord.resource_savings_vs_baseline` | per-tick or run-average | Backend |
| **Per-group delivery detail** (mode, `n_requesting`, `required_resource_mbps`, `congestion`, `achievable_mbps_by_ue`) | The raw result of one group's delivery this tick | mixed | `traffic_sim/delivery/result.py :: DeliveryResult` | per-group, per-tick | Backend computes it; **not currently persisted** — see Gap note below |

> **Gap:** `DeliveryResult` objects exist only transiently inside `Simulator.step()` (`traffic_sim/simulation/simulator.py`, the `delivery_results` list) — they are consumed by `KPIEngine.record_tick()` into the aggregate `KPIRecord` and then discarded. If the dashboard wants a live per-group breakdown (not just the tick-level aggregate `mode_breakdown` string), a future API layer needs to capture `delivery_results` per tick before it's dropped — e.g. keep the last N ticks in memory. This document flags it rather than inventing a field that doesn't exist yet.

### 3.4 AI Decision / Explainability

All from `traffic_sim/ai/agent.py :: AIDecision`, one per (cell, segment) group, per tick — produced by `AIAgent.decide()`. Also transient (see the same Gap pattern as 3.3): captured today via `AIAgent.current_tick_stats().decisions`, valid only between the end of a tick's decisions and the next `observe()` call.

| Field | Meaning | Unit | Backend/Frontend |
|---|---|---|---|
| `mode` | Final decided mode | enum | Backend |
| `confidence` | How decisively the utility scores favoured this mode (0 = tied, 1 = ≥0.25 apart) | 0–1 | Backend |
| `reason.candidate` | What the cell-coordinated ranking proposed before safety gates | string | Backend |
| `reason.utility_gap` | `|U_broadcast − U_unicast|` | utility units | Backend |
| `reason.broadcast_benefit` | `max(0, U_broadcast − U_unicast)` — the cell-controller's ranking score | utility units | Backend |
| `reason.winning_terms` | The chosen mode's utility breakdown by term (see 3.4.1) | dict | Backend |
| `reason.prediction_source` | `"xgboost"` or `"analytical"` | string | Backend |
| `reason.coordination` | `"cell_admission"` or `"independent"` (fallback path) | string | Backend |
| `reason.action` | Which safety/policy outcome fired: `initial_assignment`, `hold_already_optimal`, `switch`, `hold_buffer_unsafe`, `hold_low_confidence`, `hold_within_hysteresis`, `hold_cooldown`, `switch_escape_stall`, `hold_pcf_policy_violation` | string (enum-like) | Backend |
| `reason.mbos` | `{session_id, target_bearer, handover, corrected}` | dict | Backend |
| `reason.pcf` | `{compliant, violated_rules}` | dict | Backend |
| `utility_unicast`, `utility_broadcast` | Total utility score per mode | float | Backend |
| `predicted_unicast_mbps`, `predicted_broadcast_mbps` | XGBoost's (or the analytical fallback's) predicted achievable rate per mode | Mbps | Backend |
| `prediction_source` | Same as `reason.prediction_source`, top-level for convenience | string | Backend |
| `inference_latency_ms` | Time this one prediction took | ms | Backend |
| `switched` | Did the mode actually change this tick | bool | Backend |
| `session_id` | MBOS session id, `"{cell_id}:{segment_id}"` | string | Backend |
| `target_bearer` | `"cellular"` or `"broadcast"` | string | Backend |
| `handover` | Bearer changed vs. last tick | bool | Backend |
| `pcf_compliant` | Did the decision pass PCF unmodified | bool | Backend |
| `pcf_violated_rules` | Which PCF rules fired (`priority_buffer_margin`, `confidence_floor`) | list[string] | Backend |

**3.4.1 — `winning_terms` detail** (`traffic_sim/ai/utility.py :: UtilityBreakdown.explain()`), each already weight-multiplied and in [0,1] roughly:

| Term | Meaning |
|---|---|
| `quality` | Can this mode sustain the segment's bitrate? |
| `resource` | Resource saved vs. cell capacity, earned only in proportion to delivery quality |
| `buffer` | Margin before a stall, after this tick |
| `popularity` | Benefit of one shared copy serving many UEs (broadcast only, scales with group size) |
| `mobility` | Penalty for a dispersing group (broadcast only) |
| `switch` | Flat penalty if this mode differs from the current one |

### 3.5 Radio Conditions

| Field | Meaning | Unit | Source | Granularity | Backend/Frontend |
|---|---|---|---|---|---|
| `mean_sinr_db` | Average signal quality UEs actually experienced | dB | `KPIRecord.mean_sinr_db` (via `network/sinr.py :: sinr_to_db`) | per-tick or run-average | Backend |
| `sinr_linear` | Per-UE SINR (linear) | ratio | `UE.sinr_linear` | per-UE, per-tick | Backend |
| `mean_achievable_mbps` | Average actual delivery rate achieved | Mbps | `KPIRecord.mean_achievable_mbps` | per-tick or run-average | Backend |
| `achievable_mbps_by_ue` | Per-UE achievable rate for this delivery | Mbps | `DeliveryResult.achievable_mbps_by_ue` | per-UE, per-group, per-tick | Backend computes it; same persistence gap as 3.3 |

> Raw Shannon capacity (bits/s theoretical ceiling) is computed transiently inside `traffic_sim/network/radio.py :: compute_ue_radio_state()` (`RadioState.shannon_mbps`) during feature extraction and delivery, but is not stored anywhere after that tick. If the dashboard wants to show "theoretical ceiling vs. achieved," that's a small backend addition (persist `RadioState` alongside the UE/DeliveryResult for the tick), not something to fake on the frontend.

### 3.6 Buffer / Playback / QoE

| Field | Meaning | Unit | Source | Granularity | Backend/Frontend |
|---|---|---|---|---|---|
| `buffer_seconds` | Playable content stored | seconds | `UE.buffer_seconds` | per-UE, per-tick | Backend |
| `mean_buffer_seconds` | Average across all UEs | seconds | `KPIRecord.mean_buffer_seconds` | per-tick/aggregate | Backend |
| `min_group_buffer_seconds` | Low-percentile (not mean) group buffer — a safety indicator, deliberately not averaged | seconds | `KPIRecord.min_group_buffer_seconds` (`simulation/buffer.py :: group_safety_buffer`) | per-tick/aggregate | Backend |
| `buffer_starvation_rate` | Fraction of UEs with an empty buffer | 0–1 | `KPIRecord.buffer_starvation_rate` | per-tick/aggregate | Backend |
| `service_continuity` | `1 − (interruption time / total playback time)` — the single continuity metric, defined purely from actual stalls | 0–1 | `KPIRecord.service_continuity` (`kpi/reliability.py`) | per-tick/aggregate | Backend |
| `rebuffering_events` / `total_rebuffering_events` | Stall episode count | count | `KPIRecord.rebuffering_events` / `KPIEngine.summary()["total_rebuffering_events"]` | per-tick / run-total | Backend |
| `mean_qoe` | Average Quality of Experience score (weighted: buffer health + no-starvation) | 0–1 | `KPIRecord.mean_qoe` (`kpi/qoe.py`) | per-tick/aggregate | Backend |
| `qoe_state` | Per-UE `good`/`degraded`/`starved` | enum | `UE.qoe_state` | per-UE, per-tick | Backend |
| `successful_delivery_ratio` | Requested segments actually delivered | 0–1 | `KPIRecord.successful_delivery_ratio` | per-tick/aggregate | Backend |
| `mean_spectrum_efficiency` | Bits delivered per Hz of bandwidth used | bits/s/Hz | `KPIRecord.mean_spectrum_efficiency` | per-tick/aggregate | Backend |

### 3.7 AI Switching History

| Field | Meaning | Unit | Source | Granularity | Backend/Frontend |
|---|---|---|---|---|---|
| `switch_count` | Groups that changed mode this tick | count | `KPIRecord.switch_count` (populated only when the policy is `ai`) | per-tick | Backend |
| `total_switches` | Sum across the run | count | `KPIEngine.summary()["total_switches"]` | run-total | Backend |
| `ai_inference_latency_ms` | Mean per-group inference time this tick | ms | `KPIRecord.ai_inference_latency_ms` | per-tick | Backend |
| `mean_inference_latency_ms` | Run-average of the above | ms | `KPIEngine.summary()["mean_inference_latency_ms"]` | run-average | Backend |
| `AIDecision.switched` | Did *this specific group* switch this tick | bool | `traffic_sim/ai/agent.py` | per-group, per-tick | Backend |
| `AIDecision.handover` | Did the bearer actually change (MBOS-confirmed) | bool | same | per-group, per-tick | Backend |
| Switch timeline (a sequence like `U → U → B → B → U`) | Built by plotting `mode_breakdown` (3.1) or per-group `mode`/`switched` across consecutive `KPIRecord`s/`AIDecision`s over time | — | Derived by the frontend **purely by plotting backend-provided per-tick values in sequence** — no new computation, just charting | timeseries | Backend supplies each point; frontend arranges them on a timeline |

### 3.8 AI vs. Baseline Comparison

From `traffic_sim/experiments/compare.py :: run_comparison()` — runs `always-unicast`, `always-broadcast`, `rule-by-group-size`, and `ai` on matched seeds/scenario and averages each policy's `KPIEngine.summary()`.

| Field | Meaning | Unit | Source | Granularity | Backend/Frontend |
|---|---|---|---|---|---|
| `policy` | Which policy this row is | string | `ComparisonRow.policy` | per-policy | Backend |
| `summary.*` | Every `KPIEngine.summary()` key (3.3/3.5/3.6/3.7's aggregate fields), averaged across the seeds used | mixed | `ComparisonRow.summary` | per-policy, run-averaged | Backend |
| Suggested headline columns | `resource_savings_vs_baseline`, `service_continuity`, `mean_qoe`, `mean_congestion`, `buffer_starvation_rate`, `mean_achievable_mbps`, `total_switches`, `mean_inference_latency_ms` | mixed | `traffic_sim/experiments/compare.py :: REPORTED_KPIS` — this is literally the backend's own recommended column set | per-policy | Backend picks the columns; frontend just renders them |

---

## Part 4 — REST API Contract (Specification)

**This is a proposed contract for a future implementation — no server exists yet.** Endpoints return JSON built directly from the dataclasses cited in Part 3 (most already support `dataclasses.asdict()`-style serialization, as `KPIEngine.to_records()` demonstrates). Nothing here requires the frontend to compute a KPI itself.

### `GET /api/config`
Returns the active `SimConfig` (`SimConfig.to_dict()` — already implemented, just needs to be served).
```json
{
  "num_cells": 3, "num_ues": 100, "num_segments": 10,
  "cellular_bandwidth_hz": 10000000.0, "broadcast_bandwidth_hz": 6000000.0,
  "cell_capacity_mbps": 100.0, "broadcast_cell_capacity_mbps": 60.0,
  "hysteresis_margin": 0.05, "cooldown_period_s": 5.0,
  "min_decision_confidence": 0.15, "pcf_priority_threshold": 1,
  "pcf_priority_margin_multiplier": 1.5, "...": "...(every SimConfig field)"
}
```

### `POST /api/simulations`
Start a run. Body:
```json
{ "scenario": "flash_crowd", "mode_selector": "ai", "ticks": 300, "seed": 11 }
```
`scenario` ∈ `{"uniform", "flash_crowd"}`; `mode_selector` ∈ `traffic_sim/simulation/mode_selectors.py::SELECTOR_FACTORY` keys (`"always-unicast"`, `"always-broadcast"`, `"random"`, `"rule-by-group-size"`, `"ai"`). Response: `{ "simulation_id": "sim_abc123", "status": "running" }`.

### `GET /api/simulations/{id}/status`
```json
{ "simulation_id": "sim_abc123", "status": "running", "current_tick": 142, "total_ticks": 300 }
```

### `GET /api/simulations/{id}/kpis/latest`
The most recent `KPIRecord`, `dataclasses.asdict()`-serialized (field list: Part 3.1/3.3/3.5/3.6/3.7).
```json
{
  "tick": 142, "sim_time_s": 142.0, "mode_breakdown": "unicast=6,broadcast=2",
  "mean_congestion": 0.31, "resource_savings_vs_baseline": 0.74,
  "mean_buffer_seconds": 8.2, "service_continuity": 0.97, "mean_qoe": 0.61,
  "switch_count": 1, "ai_inference_latency_ms": 0.71, "...": "...(all KPIRecord fields)"
}
```

### `GET /api/simulations/{id}/kpis/history?from_tick=&to_tick=`
List of `KPIRecord` (i.e. `KPIEngine.to_records()`), for building timeline charts (3.7).

### `GET /api/simulations/{id}/kpis/summary`
`KPIEngine.summary()` verbatim — the run-averaged aggregates used throughout Part 3.

### `GET /api/simulations/{id}/cells`
List of current `Cell` states (Part 3.2), each including the computed `congestion`.

### `GET /api/simulations/{id}/ues?cell_id=&limit=&offset=`
Paginated list of current `UE` states (Part 3.2) — with ~100 UEs, pagination keeps payloads small; `cell_id` narrows to one cell for the UE-map view.

### `GET /api/simulations/{id}/decisions/latest`
List of this tick's `AIDecision` objects (Part 3.4) — **only meaningful when `mode_selector == "ai"`**; empty list for fixed-rule policies. *(Requires the small backend addition noted under 3.3/3.4 — capturing `AIAgent.current_tick_stats().decisions` per tick instead of discarding it.)*
```json
[
  {
    "mode": "broadcast", "confidence": 0.83,
    "reason": {
      "candidate": "broadcast", "utility_gap": 0.21, "broadcast_benefit": 0.21,
      "winning_terms": {"quality": 1.0, "resource": 0.9, "buffer": 0.8, "popularity": 0.75, "mobility": 0.0, "switch": 0.0},
      "prediction_source": "xgboost", "coordination": "cell_admission", "action": "switch",
      "mbos": {"session_id": "cell_0:seg_2", "target_bearer": "broadcast", "handover": true, "corrected": false},
      "pcf": {"compliant": true, "violated_rules": []}
    },
    "utility_unicast": 0.31, "utility_broadcast": 0.52,
    "predicted_unicast_mbps": 1.2, "predicted_broadcast_mbps": 2.4,
    "prediction_source": "xgboost", "inference_latency_ms": 0.68, "switched": true,
    "session_id": "cell_0:seg_2", "target_bearer": "broadcast", "handover": true,
    "pcf_compliant": true, "pcf_violated_rules": []
  }
]
```

### `GET /api/comparison?scenario=&ticks=&seeds=`
Runs `run_comparison()` (Part 3.8) and returns the per-policy rows.
```json
[
  { "policy": "always-unicast", "summary": { "resource_savings_vs_baseline": 0.0, "service_continuity": 0.71, "...": "..." } },
  { "policy": "always-broadcast", "summary": { "resource_savings_vs_baseline": 0.84, "service_continuity": 0.44, "...": "..." } },
  { "policy": "rule-by-group-size", "summary": { "resource_savings_vs_baseline": 0.30, "service_continuity": 0.83, "...": "..." } },
  { "policy": "ai", "summary": { "resource_savings_vs_baseline": 0.74, "service_continuity": 0.87, "...": "..." } }
]
```
Note: this is a batch computation (up to 4 policies × several seeds × N ticks run synchronously) — see the transport recommendation below for why this endpoint is polled, not streamed.

---

## Part 5 — Live Update Transport Recommendation

**For a running single simulation (Sections 1–7 of the dashboard):** use a **WebSocket** (or Server-Sent Events, if the eventual backend framework makes that simpler) that pushes one message per tick — `{ "kpi": <KPIRecord>, "decisions": [<AIDecision>...] }` — while a simulation with `mode_selector=ai` is running. Ticks are cheap (the AI inference latency measured live is well under a millisecond per group; a full tick is a handful of milliseconds), so a 300-tick run can push tens of updates per second — a natural fit for a persistent stream rather than the frontend re-polling a REST endpoint every few milliseconds. Fall back to **REST polling** of `/kpis/latest` (2–5× per second) for a first cut or if a WebSocket layer isn't available yet — perfectly usable, just less smooth for the closed-loop "watch it adapt" narrative the dashboard should tell.

**For `/api/comparison` (Section 8):** plain **REST polling** of a job-status pattern (`POST` to start, `GET .../status` every second or two until `"status": "done"`) is the right fit — it's a short, one-shot batch computation (a handful of seconds), not a continuous stream, and doesn't need socket infrastructure.

---

## Part 6 — Design Recommendations for Manus

These are recommendations, not specifications — Manus should feel free to adapt them to good UI judgment. No backend changes are implied by anything in this section.

**Overall layout.** A left-hand or top persistent bar for run controls (scenario, mode-selector, start/stop, current tick) and global KPIs (service continuity, resource savings, QoE — the three numbers a reviewer should be able to read in one glance). The main area tabs or scrolls through the 8 sections in the order they were defined in Part 3 — that order already reads as a story: overview → what the network looks like → what mode is being used → why the AI chose it → why the radio made that possible → what the user actually experienced → how it changed over time → how it compares to not having an AI at all.

**Cards/charts that matter most:**
- A **single headline row** of 3–4 big stat tiles: service continuity, resource savings vs. always-unicast, mean QoE, current mode split (unicast % / broadcast %). This is what a judge or reviewer looks at in the first three seconds.
- A **live switch timeline** (Section 7): a horizontal strip per active group (or an aggregated stacked-area chart of unicast vs. broadcast group counts over time) — this is the single best visual proof that the system is *dynamically adapting*, not applying a fixed rule. Mark handovers distinctly.
- An **AI decision explainability panel** (Section 4): when a user clicks a group/segment, show its `winning_terms` as a small horizontal bar breakdown (quality/resource/buffer/popularity/mobility/switch) plus the plain-English `action` string and, when relevant, the PCF `violated_rules` — this is what makes the AI's reasoning inspectable rather than a black box.
- An **AI vs. baseline comparison table/chart** (Section 8): a grouped bar chart, one group per KPI (resource saved, continuity, QoE), one bar per policy — this is the project's actual thesis made visible: the AI should sit closer to `always-broadcast`'s resource savings while staying much closer to `always-unicast`'s continuity than `always-broadcast` does.

**Visualizing cells/UEs.** A simple 2D scatter plot using `x_km`/`y_km` (synthetic coordinates, not a real map — don't imply real geography) is enough: cells as labeled circles sized by capacity or shaded by `congestion`, UEs as small dots colored by `delivery_mode` (two colors, unicast vs. broadcast) and optionally by `qoe_state` (a third visual channel, e.g. dot opacity or a thin ring for `starved`). Avoid trying to render a literal "signal strength heatmap" — the radio model is intentionally simplified (Part 1), and implying more physical realism than exists would be misleading.

**Visualizing AI decisions and switching.** Use one consistent color pair for unicast vs. broadcast everywhere in the dashboard (map view, timeline, comparison chart) — consistency here does more for clarity than any single chart's cleverness. A switch event (`AIDecision.switched`) is worth a brief visual pulse/highlight on whichever group/cell it happened in, so the "adapting live" story is felt, not just read from a table.

**Suggested visual style.** A dark, technical, telecom/NOC-dashboard aesthetic reads as credible for this domain: dark background, a small set of saturated accent colors reserved specifically for state (e.g., one hue for unicast, a contrasting hue for broadcast, a third for alerts/violations), monospace or semi-condensed numerals for KPI tiles, generous whitespace between sections rather than dense borders. Avoid skeuomorphic gauges/dials — flat stat tiles and clean line/area/bar charts fit a technical audience better and hold up under the 8-section density this dashboard needs.

**Keeping it impressive but not cluttered.** Default to showing the **headline row + switch timeline + cell/UE map** on first load; put radio conditions, per-UE buffer detail, and the full comparison table behind a tab or a scroll — reviewers should see the thesis (AI adapts, and it's better than either fixed extreme) in seconds, with the depth available but not demanded. Every number should be traceable to Part 3 of this document — resist the urge to add a metric that "looks nice" but isn't backed by a real backend field; that's exactly the duplication this handoff is written to prevent.
