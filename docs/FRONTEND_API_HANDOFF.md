# Frontend / API Handoff — AI-Enabled Smart Data Traffic Switching

**Audience:** Manus (frontend/dashboard implementer).

**Status: the API in Part 4 is implemented and ready to connect to.** Start it with `pip install -e '.[api]'` then `python -m traffic_sim.api`, and browse the live Swagger docs at `http://127.0.0.1:8000/docs`. It serves real data from real simulation runs — nothing is mocked. The WebSocket in Part 5 is implemented too. No frontend code exists yet; that is Manus's job, and Parts 3 and 6 are written to support it.

Every field named below is cited to a real file/function/class in the repository. Nothing here is invented. Where a value is only available for the current tick rather than historically, that is stated plainly rather than glossed over.

---

## Quick Start (60 seconds)

```bash
pip install -e '.[api]'
python -m traffic_sim.api            # http://127.0.0.1:8000  (Swagger: /docs)
```

```bash
# 1. start a watchable AI run (0.2s per tick)
curl -X POST http://127.0.0.1:8000/api/simulations \
  -H 'Content-Type: application/json' \
  -d '{"scenario":"flash_crowd","mode_selector":"ai","ticks":120,"seed":11,"tick_interval_s":0.2}'
# -> {"simulation_id":"sim_abc123","status":"running", ...}

# 2. watch it live over the WebSocket
#    ws://127.0.0.1:8000/api/simulations/sim_abc123/live

# 3. or read any snapshot over REST
curl http://127.0.0.1:8000/api/simulations/sim_abc123/kpis/latest
curl http://127.0.0.1:8000/api/simulations/sim_abc123/decisions/history?limit=20
curl 'http://127.0.0.1:8000/api/comparison?scenario=flash_crowd&ticks=20&seeds=11&seeds=22'
```

Full endpoint reference in **Part 4**; live-update details in **Part 5**; what every field means and where it comes from in **Part 3**.

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
| **Per-group delivery detail** (mode, `n_requesting`, `required_resource_mbps`, `congestion`, `sinr_by_ue`, `achievable_mbps_by_ue`, `content_received_seconds_by_ue`, `delivered_ue_ids`) | The raw result of one group's delivery this tick | mixed | `traffic_sim/delivery/result.py :: DeliveryResult`, served by `GET /api/simulations/{id}/groups/latest` | per-group, per-tick | Backend |

> **Current-tick only.** `DeliveryResult` objects are transient inside `Simulator.step()` — consumed by `KPIEngine.record_tick()` into the aggregate `KPIRecord`, then discarded. The API layer (`traffic_sim/api/runner.py`) captures them as they pass through `record_tick` and keeps **the latest tick only**, which is what `/groups/latest` serves. There is no per-group delivery *history*: for anything older than the current tick, use the tick-level aggregates in `KPIRecord` (`mode_breakdown`, `total_required_resource_mbps`, …) via `/kpis/history`. If a historical per-group breakdown is ever needed, the capture buffer in `SimulationRunner` is where it would be widened.

### 3.4 AI Decision / Explainability

**Deliberately simplified.** The dashboard shows a *conclusion*, not a derivation: the final delivery mode, how confident the agent was, and one plain-English sentence. The internal chain — the two XGBoost rate predictions, the per-term utility breakdown, the cell-admission ranking, the MBOS/PCF internals — still runs and is still tested, but it is **not exposed through the API** and should not appear on screen. It is engineering detail, not something a viewer can act on.

Served by `GET /api/simulations/{id}/decisions/latest` (and included on every WebSocket tick). Derived from `traffic_sim/ai/agent.py :: AIDecision`, one per (cell, segment) group per tick; the sentence is rendered by `traffic_sim/api/explain.py :: explain_decision()`, which is presentation only — it adds no logic and no thresholds of its own.

| Field | Meaning | Unit | Granularity | Backend/Frontend |
|---|---|---|---|---|
| `cell_id` | Which cell this decision belongs to | string | per-group, per-tick | Backend |
| `group_id` | Which content segment/group | string | per-group, per-tick | Backend |
| `selected_mode` | **The FINAL mode, after MBOS coordination and PCF enforcement** — never the agent's pre-policy candidate | `"unicast"` / `"broadcast"` | per-group, per-tick | Backend |
| `previous_mode` | The mode in force immediately before this decision (`null` on a group's first decision) | string or null | per-group, per-tick | Backend |
| `switched` | Did the mode actually change this tick | bool | per-group, per-tick | Backend |
| `confidence` | How decisively the two options separated (0 = tied, 1 = clearly apart) | 0–1 | per-group, per-tick | Backend |
| `explanation` | One human-readable sentence saying what happened and why | string | per-group, per-tick | Backend |
| `bearer` | The physical bearer carrying it: `"cellular"` or `"broadcast"` | string | per-group, per-tick | Backend |
| `policy_compliant` | `false` when PCF overrode the agent's choice | bool | per-group, per-tick | Backend |

**The nine explanations**, one per outcome the agent can reach:

| Situation | Sentence shown |
|---|---|
| First decision for a group | "Started on {mode} — first decision for this group." |
| Current mode still best | "Staying on {mode} — it is still the best option." |
| Genuine improvement | "Switched to {mode} — it now delivers this content better." |
| Rescuing stalled playback | "Switched to {mode} — playback was stalling on {previous}." |
| Buffers too low to risk it | "Staying on {mode} — buffers are too low to risk switching." |
| Options too close to call | "Staying on {mode} — the two options are too close to call." |
| Gain too small to bother | "Staying on {mode} — the gain from switching is too small to be worth it." |
| Switched too recently | "Staying on {mode} — it switched too recently to change again." |
| Blocked by network policy | "Staying on {mode} — network policy blocked the change: {plain-English rule}." |

A decision panel showing `selected_mode`, `confidence` and `explanation` is the whole story a viewer needs. An unrecognised reason code degrades to a readable fallback sentence rather than erroring.

### 3.5 Radio Conditions

| Field | Meaning | Unit | Source | Granularity | Backend/Frontend |
|---|---|---|---|---|---|
| `mean_sinr_db` | Average signal quality UEs actually experienced | dB | `KPIRecord.mean_sinr_db` (via `network/sinr.py :: sinr_to_db`) | per-tick or run-average | Backend |
| `sinr_linear` | Per-UE SINR (linear) | ratio | `UE.sinr_linear` | per-UE, per-tick | Backend |
| `mean_achievable_mbps` | Average actual delivery rate achieved | Mbps | `KPIRecord.mean_achievable_mbps` | per-tick or run-average | Backend |
| `achievable_mbps_by_ue` | Per-UE achievable rate for this delivery | Mbps | `DeliveryResult.achievable_mbps_by_ue`, via `/groups/latest` | per-UE, per-group, per-tick | Backend (current tick only — see 3.3) |
| `sinr_by_ue` | Per-UE SINR for this delivery | ratio | `DeliveryResult.sinr_by_ue`, via `/groups/latest` | per-UE, per-group, per-tick | Backend (current tick only — see 3.3) |

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

**Tick-level aggregate (from `KPIRecord`, every tick, any policy):**

| Field | Meaning | Unit | Source | Granularity | Backend/Frontend |
|---|---|---|---|---|---|
| `switch_count` | Groups that changed mode this tick | count | `KPIRecord.switch_count` (populated only when the policy is `ai`) | per-tick | Backend |
| `total_switches` | Sum across the run | count | `KPIEngine.summary()["total_switches"]` | run-total | Backend |
| `ai_inference_latency_ms` | Mean per-group inference time this tick | ms | `KPIRecord.ai_inference_latency_ms` | per-tick | Backend |
| `mean_inference_latency_ms` | Run-average of the above | ms | `KPIEngine.summary()["mean_inference_latency_ms"]` | run-average | Backend |

**Per-group decision history — now persisted** (`traffic_sim/ai/agent.py :: DecisionRecord`, produced by `AIAgent.decision_history(limit=None)`; only present when the policy is `ai`). Unlike the live-only `AIDecision`/`current_tick_stats()` view in 3.4, this is a bounded, in-memory rolling window (newest `config.decision_history_size` rows, default 500, oldest dropped first) that survives past the tick it was recorded on — populated once per tick inside `AIAgent.observe()`, after that tick's real `tick`/`sim_time_s` are known. `selected_mode` and `mbos_bearer` are read from the already-reconciled, **post-MBOS-post-PCF** `AIDecision` fields, so this is guaranteed to be the mode that was actually delivered, never the agent's pre-PCF candidate:

| Field | Meaning | Unit | Source | Granularity | Backend/Frontend |
|---|---|---|---|---|---|
| `tick` | Which tick this decision was made on | int | `DecisionRecord.tick` (= that tick's `KPIRecord.tick`) | per-group, per-tick | Backend |
| `sim_time_s` | Simulated clock time of that tick | seconds | `DecisionRecord.sim_time_s` | per-group, per-tick | Backend |
| `cell_id` | Which cell | string | `DecisionRecord.cell_id` | per-group, per-tick | Backend |
| `group_id` | Which content segment/group (the `(cell_id, group_id)` pair is the group key used throughout the AI layer) | string | `DecisionRecord.group_id` | per-group, per-tick | Backend |
| `previous_mode` | The mode this group was in immediately before this decision | `"unicast"` / `"broadcast"` / `null` | `DecisionRecord.previous_mode` | per-group, per-tick | Backend |
| `selected_mode` | The FINAL mode, after MBOS coordination and PCF policy enforcement | `"unicast"` / `"broadcast"` | `DecisionRecord.selected_mode` | per-group, per-tick | Backend |
| `utility_unicast`, `utility_broadcast` | Total utility score computed for each mode this tick | float | `DecisionRecord.utility_unicast` / `utility_broadcast` | per-group, per-tick | Backend |
| `broadcast_benefit` | `max(0, utility_broadcast − utility_unicast)` — the cell-controller's ranking score | float | `DecisionRecord.broadcast_benefit` | per-group, per-tick | Backend |
| `confidence` | How decisively the utility scores favoured `selected_mode`'s pre-PCF candidate | 0–1 | `DecisionRecord.confidence` | per-group, per-tick | Backend |
| `decision_reason` | Which safety/policy outcome fired: `initial_assignment`, `hold_already_optimal`, `switch`, `hold_buffer_unsafe`, `hold_low_confidence`, `hold_within_hysteresis`, `hold_cooldown`, `switch_escape_stall`, `hold_pcf_policy_violation` | string (enum-like) | `DecisionRecord.decision_reason` | per-group, per-tick | Backend |
| `mbos_bearer` | The physical bearer actually used: `"cellular"` or `"broadcast"` | string | `DecisionRecord.mbos_bearer` | per-group, per-tick | Backend |
| `pcf_result` | `"compliant"`, or `"reverted:<rule_name>[,<rule_name>...]"` when PCF overrode the mode | string | `DecisionRecord.pcf_result` | per-group, per-tick | Backend |
| `inference_latency_ms` | How long this one group's prediction took | ms | `DecisionRecord.inference_latency_ms` | per-group, per-tick | Backend |
| `explanation` | The same plain sentence as 3.4 — what happened and why | string | `traffic_sim/api/explain.py :: explain_record()`, added by the API when serializing the row | per-group, per-tick | Backend |

Switch timeline (a sequence like `U → U → B → B → U`), per group or per cell: build it by plotting consecutive `decision_history()` rows' `selected_mode` in `tick` order — no new computation on the frontend, just charting backend-provided values in sequence.

### 3.8 AI vs. Baseline Comparison

From `traffic_sim/experiments/compare.py :: run_comparison()` — runs `always-unicast`, `always-broadcast`, `rule-by-group-size`, and `ai` on matched seeds/scenario and averages each policy's `KPIEngine.summary()`.

| Field | Meaning | Unit | Source | Granularity | Backend/Frontend |
|---|---|---|---|---|---|
| `policy` | Which policy this row is | string | `ComparisonRow.policy` | per-policy | Backend |
| `summary.*` | Every `KPIEngine.summary()` key (3.3/3.5/3.6/3.7's aggregate fields), averaged across the seeds used | mixed | `ComparisonRow.summary` | per-policy, run-averaged | Backend |
| Suggested headline columns | `resource_savings_vs_baseline`, `service_continuity`, `mean_qoe`, `mean_congestion`, `buffer_starvation_rate`, `mean_achievable_mbps`, `total_switches`, `mean_inference_latency_ms` | mixed | `traffic_sim/experiments/compare.py :: REPORTED_KPIS` — this is literally the backend's own recommended column set | per-policy | Backend picks the columns; frontend just renders them |

---

## Part 4 — REST API (Implemented)

**This API is implemented and running-ready** — `traffic_sim/api/` (FastAPI), covered by `tests/test_api.py`. Endpoints return JSON built directly from the dataclasses cited in Part 3. Nothing here requires the frontend to compute a KPI itself.

### Running it

```bash
pip install -e '.[api]'        # fastapi + uvicorn (optional extra)
python -m traffic_sim.api      # -> http://127.0.0.1:8000
```

Flags: `--host`, `--port`, `--reload`. Interactive Swagger docs are auto-generated at **`http://127.0.0.1:8000/docs`** — the fastest way to explore every endpoint and see live response shapes without writing a line of frontend code.

**CORS is wide open** (`allow_origins=["*"]`), so a dashboard served from any origin/port can call this directly in development. That is deliberate for a local PoC serving simulated data and holding no credentials; a deployed instance should name its allowed origins instead.

### How it relates to the simulator

The API never re-implements simulation logic. It builds simulators through the existing `traffic_sim.main.build_simulator()` seam and advances them one `Simulator.step()` at a time on a background thread, which is what makes a run observable while it is still in flight. A parity test (`test_api_run_matches_a_direct_simulator_run_exactly`) asserts that the same seed/scenario/tick-count through the API produces KPIs identical to a direct `build_simulator(...).run(...)`, so the serving layer provably observes without perturbing.

### Endpoint summary

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness check |
| `GET /api/config` | Every `SimConfig` field |
| `GET /api/scenarios` | Valid `scenario` / `mode_selector` values and limits |
| `POST /api/simulations` | Start a run |
| `GET /api/simulations` | List all runs held by the server |
| `GET /api/simulations/{id}/status` | Progress and lifecycle state |
| `POST /api/simulations/{id}/pause` | Freeze a run between ticks |
| `POST /api/simulations/{id}/resume` | Continue a paused run |
| `POST /api/simulations/{id}/stop` | Halt a run early |
| `DELETE /api/simulations/{id}` | Discard a run |
| `GET /api/simulations/{id}/kpis/latest` | Newest `KPIRecord` |
| `GET /api/simulations/{id}/kpis/history` | All `KPIRecord`s (tick-range filterable) |
| `GET /api/simulations/{id}/kpis/summary` | Run-averaged aggregates |
| `GET /api/simulations/{id}/cells` | Live cell state |
| `GET /api/simulations/{id}/ues` | Live UE state (paginated, cell-filterable) |
| `GET /api/simulations/{id}/groups/latest` | Per-group delivery detail, current tick |
| `GET /api/simulations/{id}/decisions/latest` | Full-detail AI decisions, current tick |
| `GET /api/simulations/{id}/decisions/history` | Persisted switching history |
| `GET /api/comparison` | AI-vs-baseline table |
| `WS /api/simulations/{id}/live` | One push message per tick |

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
Start a run (HTTP 201). Body — every field optional, defaults shown:
```json
{
  "scenario": "flash_crowd", "mode_selector": "ai", "ticks": 120, "seed": null,
  "tick_interval_s": 0.0, "config_overrides": null
}
```
- `scenario` ∈ `{"uniform", "flash_crowd"}`; `mode_selector` ∈ `SELECTOR_FACTORY` keys (`"always-unicast"`, `"always-broadcast"`, `"random"`, `"rule-by-group-size"`, `"ai"`). Fetch both lists from `/api/scenarios` rather than hardcoding them.
- **`tick_interval_s`** (added beyond the original spec) — seconds to pause between ticks. `0.0` runs flat out, best when you just want a finished run to chart. `0.1`–`0.5` makes the run *watchable*: use it for the live "watch the AI adapt" view.
- **`config_overrides`** (added beyond the original spec) — any `SimConfig` field, e.g. `{"broadcast_cell_capacity_mbps": 4.0}` to demo a starved broadcast carrier. Validated by `SimConfig` itself; unknown fields or invalid values return 400.

Response is the same shape as `/status` below. Invalid selector/scenario/override → 400; out-of-range `ticks`/`tick_interval_s` → 422.

### `GET /api/simulations` (added beyond the original spec)
List of every run the server currently holds, each in the `/status` shape.

### `GET /api/simulations/{id}/status`
```json
{
  "simulation_id": "sim_abc123", "status": "running",
  "current_tick": 142, "total_ticks": 300,
  "scenario": "flash_crowd", "mode_selector": "ai",
  "tick_interval_s": 0.1, "seed": 11, "error": null
}
```
`status` ∈ `pending | running | paused | completed | stopped | failed`. On `failed`, `error` carries the exception text.

### `POST /api/simulations/{id}/pause` · `.../resume` (added beyond the original spec)
Freeze a live run between ticks and continue it later — the core interaction for a dashboard you can actually inspect: pause on an interesting decision, read it, resume. Both return the `/status` payload.

An open **WebSocket stays connected across a pause** (`paused` is not a terminal status); it simply stops receiving ticks until you resume. A paused run can still be stopped or deleted.

Returns **409** if the transition doesn't apply (resuming a run that isn't paused, pausing one that isn't running), so the frontend can drive the buttons off `status` and treat a 409 as "someone else already changed it".

### `POST /api/simulations/{id}/stop` · `DELETE /api/simulations/{id}` (added beyond the original spec)
Halt a run early (returns the final `/status` payload, `status: "stopped"`), or discard it entirely (`{"simulation_id": ..., "deleted": true}`).

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
List of current `Cell` states (Part 3.2), each including the computed `congestion` (a method on `Cell`, not a stored field) plus `num_connected_ues`.

### `GET /api/simulations/{id}/ues?cell_id=&limit=&offset=`
Paginated list of current `UE` states (Part 3.2); `cell_id` narrows to one cell for the UE-map view. **Differs from the original proposal:** returns an envelope, not a bare array, so the frontend can page without a second count request. `limit` defaults to 200 (max 1000). All enum fields arrive as plain strings (`"unicast"`, `"good"`, …).
```json
{ "total": 100, "limit": 200, "offset": 0, "items": [ { "ue_id": "ue_0", "...": "..." } ] }
```

### `GET /api/simulations/{id}/groups/latest` (added beyond the original spec)
Per-group `DeliveryResult`s for the most recent tick (Part 3.3) — the per-group unicast/broadcast breakdown behind the aggregate `mode_breakdown` string. Current tick only.

### `GET /api/simulations/{id}/decisions/latest`
This tick's AI decisions in the simplified dashboard shape (Part 3.4) — final mode, confidence, and one plain sentence. **Only populated when `mode_selector == "ai"`**; `[]` for fixed-rule policies, which make no explainable decision. Current tick only; for decisions across ticks use `/decisions/history` below.

The internal reasoning chain (utility terms, predicted rates, MBOS/PCF internals) is intentionally **not** in this payload — see 3.4.
```json
[
  {
    "cell_id": "cell_0", "group_id": "seg_2",
    "selected_mode": "broadcast", "previous_mode": "unicast", "switched": true,
    "confidence": 0.88,
    "explanation": "Switched to broadcast — it now delivers this content better.",
    "bearer": "broadcast", "policy_compliant": true
  }
]
```

### `GET /api/simulations/{id}/decisions/history?limit=`
The persisted per-group decision history (Part 3.7) — `AIAgent.decision_history(limit)`, already bounded server-side to `config.decision_history_size` rows; `limit` trims further for a smaller payload. `selected_mode` is always the final, post-MBOS-post-PCF mode. **Only meaningful when `mode_selector == "ai"`**; empty list for fixed-rule policies (they never populate this history — see 3.7).

Each row carries **`explanation`**, the same plain sentence as `/decisions/latest` (Part 3.4), so a history table is readable without decoding `decision_reason` codes on the frontend. For a switching-history view, `tick` + `selected_mode` + `explanation` is all you need; the utility numbers are there for optional charting, not headline display.
```json
[
  {
    "tick": 20, "sim_time_s": 20.0, "cell_id": "cell_1", "group_id": "seg_2",
    "previous_mode": "unicast", "selected_mode": "unicast",
    "utility_unicast": 0.547, "utility_broadcast": 0.731, "broadcast_benefit": 0.183,
    "confidence": 0.734, "decision_reason": "hold_cooldown",
    "mbos_bearer": "cellular", "pcf_result": "compliant", "inference_latency_ms": 0.80,
    "explanation": "Staying on unicast — it switched too recently to change again."
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
**Differs from the original proposal:** implemented as a single **synchronous** request rather than the POST-then-poll job pattern Part 5 originally suggested — measured at well under a second for `ticks=20, seeds=11&seeds=22`, which does not justify job-tracking machinery. Pass `seeds` as a repeated query parameter (`?seeds=11&seeds=22`); defaults are `scenario=flash_crowd`, `ticks=60`, `seeds=[11,22,33]`. It is CPU-bound (4 policies × N seeds × full simulations), so keep `ticks` modest (capped at 500) and show a spinner rather than blocking the whole dashboard on it.

---

## Part 5 — Live Updates (Implemented)

### `WS /api/simulations/{id}/live`

**The WebSocket is implemented.** Connect after starting a run (use a non-zero `tick_interval_s` so there is something to watch), and the server pushes **one message per simulated tick**:

```json
{
  "type": "tick",
  "simulation_id": "sim_abc123",
  "tick": 42,
  "kpi": { "tick": 42, "sim_time_s": 42.0, "mode_breakdown": "broadcast=13,unicast=4", "...": "...(all KPIRecord fields)" },
  "cells": [ { "cell_id": "cell_0", "congestion": 0.31, "current_load_mbps": 31.0, "num_connected_ues": 34, "...": "..." } ],
  "decisions": [
    { "cell_id": "cell_0", "group_id": "seg_2", "selected_mode": "broadcast",
      "previous_mode": "unicast", "switched": true, "confidence": 0.88,
      "explanation": "Switched to broadcast — it now delivers this content better.",
      "bearer": "broadcast", "policy_compliant": true }
  ],
  "groups": [ { "mode": "broadcast", "n_requesting": 12, "...": "..." } ]
}
```

One message carries everything the live view needs — the tick's KPIs (Sections 1, 3, 5, 6, 7), live per-cell load and congestion (Section 2), the AI decisions in their simplified form (Section 4), and the per-group delivery breakdown (Section 3) — so a live dashboard needs **no REST polling at all** while a run is in flight.

**UEs are the one exception** and stay on REST (`/ues`): ~100 objects per tick is real bandwidth for state that changes slowly. Poll them every second or so for the UE map; everything else arrives on the socket.

Behaviour worth knowing:
- On connect, the **current tick is replayed immediately**, so a client joining mid-run renders at once instead of waiting for the next tick.
- Every tick is delivered in order; none are dropped or coalesced.
- When the run finishes the server sends a final `{"type": "status", ...}` message (the `/status` payload, e.g. `"status": "completed"`) and closes.
- An unknown `simulation_id` yields `{"type": "error", "detail": "..."}` then a close.
- `decisions` is `[]` for fixed-rule policies — the stream shape is identical across policies, so the same client code handles baselines and the AI.
- **A pause does not close the socket.** Ticks simply stop arriving until `/resume`, so the dashboard keeps its live view intact across a pause.

Example:
```js
const ws = new WebSocket(`ws://127.0.0.1:8000/api/simulations/${simId}/live`);
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "tick") updateDashboard(msg.kpi, msg.decisions, msg.groups);
  else if (msg.type === "status") markRunFinished(msg.status);
};
```

**REST polling remains a valid fallback** — `GET /kpis/latest` a few times a second gives the same numbers, just less smoothly. Use it for a first cut, or if the frontend framework makes sockets awkward.

**For `/api/comparison` (Section 8):** plain request/response — it is a synchronous endpoint (see Part 4), fast enough at modest `ticks` that no job-polling pattern is needed.

---

## Part 6 — Design Recommendations for Manus

These are recommendations, not specifications — Manus should feel free to adapt them to good UI judgment. No backend changes are implied by anything in this section.

**Overall layout.** A left-hand or top persistent bar for run controls (scenario, mode-selector, start/stop, current tick) and global KPIs (service continuity, resource savings, QoE — the three numbers a reviewer should be able to read in one glance). The main area tabs or scrolls through the 8 sections in the order they were defined in Part 3 — that order already reads as a story: overview → what the network looks like → what mode is being used → why the AI chose it → why the radio made that possible → what the user actually experienced → how it changed over time → how it compares to not having an AI at all.

**Cards/charts that matter most:**
- A **single headline row** of 3–4 big stat tiles: service continuity, resource savings vs. always-unicast, mean QoE, current mode split (unicast % / broadcast %). This is what a judge or reviewer looks at in the first three seconds.
- A **live switch timeline** (Section 7): a horizontal strip per active group (or an aggregated stacked-area chart of unicast vs. broadcast group counts over time) — this is the single best visual proof that the system is *dynamically adapting*, not applying a fixed rule. Mark handovers distinctly.
- An **AI decision panel** (Section 4): when a user clicks a group/segment, show three things and stop there — the `selected_mode` (the mode actually delivered, after MBOS and PCF), the `confidence` as a simple meter or percentage, and the `explanation` sentence. That is the whole panel. Resist adding the utility-term breakdown or the predicted rates: the API no longer exposes them precisely because a decision panel that reads like a debugger is worse, not better, for the person looking at it. `policy_compliant: false` deserves a small badge ("policy override"), since that is the one case where the network, not the AI, had the last word.
- An **AI vs. baseline comparison table/chart** (Section 8): a grouped bar chart, one group per KPI (resource saved, continuity, QoE), one bar per policy — this is the project's actual thesis made visible: the AI should sit closer to `always-broadcast`'s resource savings while staying much closer to `always-unicast`'s continuity than `always-broadcast` does.

**Visualizing cells/UEs.** A simple 2D scatter plot using `x_km`/`y_km` (synthetic coordinates, not a real map — don't imply real geography) is enough: cells as labeled circles sized by capacity or shaded by `congestion`, UEs as small dots colored by `delivery_mode` (two colors, unicast vs. broadcast) and optionally by `qoe_state` (a third visual channel, e.g. dot opacity or a thin ring for `starved`). Avoid trying to render a literal "signal strength heatmap" — the radio model is intentionally simplified (Part 1), and implying more physical realism than exists would be misleading.

**Visualizing AI decisions and switching.** Use one consistent color pair for unicast vs. broadcast everywhere in the dashboard (map view, timeline, comparison chart) — consistency here does more for clarity than any single chart's cleverness. A switch event (`switched: true`) is worth a brief visual pulse/highlight on whichever group/cell it happened in, so the "adapting live" story is felt, not just read from a table. In the switching-history table, `explanation` is the column that carries the story — lead with it rather than with the raw `decision_reason` code.

**Suggested visual style.** A dark, technical, telecom/NOC-dashboard aesthetic reads as credible for this domain: dark background, a small set of saturated accent colors reserved specifically for state (e.g., one hue for unicast, a contrasting hue for broadcast, a third for alerts/violations), monospace or semi-condensed numerals for KPI tiles, generous whitespace between sections rather than dense borders. Avoid skeuomorphic gauges/dials — flat stat tiles and clean line/area/bar charts fit a technical audience better and hold up under the 8-section density this dashboard needs.

**Keeping it impressive but not cluttered.** Default to showing the **headline row + switch timeline + cell/UE map** on first load; put radio conditions, per-UE buffer detail, and the full comparison table behind a tab or a scroll — reviewers should see the thesis (AI adapts, and it's better than either fixed extreme) in seconds, with the depth available but not demanded. Every number should be traceable to Part 3 of this document — resist the urge to add a metric that "looks nice" but isn't backed by a real backend field; that's exactly the duplication this handoff is written to prevent.
