# AI-Enabled Smart Data Traffic Switching

An end-to-end simulation of an AI-driven telecom network that decides, for every content
request, whether to deliver it over **cellular unicast** or a **shared broadcast carrier** —
and adapts that decision continuously as network conditions change.

Built from scratch in Python, with a trained XGBoost decision layer, a FastAPI service, and a
live React dashboard.

---

## The problem

When many people in the same area want the same content at the same time — a live match, a
breaking-news stream — the network sends each of them a private copy. The same bytes go out
over and over, and capacity runs out long before demand does.

A shared broadcast transmission fixes that: one transmission, any number of listeners, for the
cost of a single stream. But broadcast is not free and not always better:

- The broadcast carrier is a **fixed-size shared pipe**. A few streams fit comfortably; too
  many, and every listener gets a thin, unwatchable slice.
- A broadcast must be sized for the **worst-placed device** in the group, so one user with a
  weak signal limits everyone.
- Groups **disperse**. Devices move, change cells, and stop watching.

So the right answer changes from moment to moment and from group to group. A fixed rule like
"broadcast once more than N users want it" gets this wrong often, because group size is only
one of the factors that matter.

**This project treats that choice as a prediction problem and learns to make it.**

---

## How it works

The simulator advances in discrete ticks. Every tick, devices move, request content, and are
grouped by *(cell, content segment)*. Each group then flows through this pipeline:

```
  live network state
         │
         ▼
  feature extraction ─────────── demand, buffer levels, signal quality,
         │                       mobility risk, capacity, prior congestion
         ▼
  XGBoost prediction ─────────── one model per delivery mode, each predicting
         │                       the delivery rate that mode would achieve
         ▼
  utility scoring ────────────── named, weighted terms turn each predicted
         │                       outcome into one comparable score
         ▼
  cell-level admission ───────── rank every candidate group by how much it
         │                       would gain from broadcast, then fill the
         │                       shared carrier highest-gain-first
         ▼
  safety gates ───────────────── buffer margin, confidence floor,
         │                       hysteresis, cooldown, stall escape
         ▼
  MBOS coordination ──────────── assign a session and a physical bearer,
         │                       flag handovers
         ▼
  policy enforcement (PCF) ───── independently re-validate the decision;
         │                       may revert it
         ▼
  delivery ───────────────────── the chosen mode actually transmits
         │
         ▼
  measurement ────────────────── per-tick KPIs
         │
         └──────── feedback into the next tick's features ──────────┐
                                                                    │
         ◄──────────────────────────────────────────────────────────┘
```

### The physics pushes back

Delivery is not scripted to favour either mode. Content moves through a full radio chain:

```
distance → path loss → received power → SINR → Shannon capacity
        → achievable rate → seconds of content → playback buffer → stall / no stall
```

Unicast devices **share** the cell's capacity, so throughput degrades as a cell fills.
A broadcast transmission is **not** divided by group size — that is the physical reason it wins
under a crowd — but it **is** divided among concurrent broadcast streams, which is what stops
"broadcast everything" from being trivially correct.

Because every group's mode is decided *before* any delivery happens, moving a group to
broadcast genuinely frees cellular capacity for the devices that stay on unicast.

---

## Design principles

The decision layer is built to be inspectable rather than opaque.

**The model predicts outcomes, not decisions.** Two regressors estimate *what each mode would
achieve*; a separate utility function decides *which outcome is preferable*. A single
classifier emitting "unicast" or "broadcast" would have been simpler, and would have buried the
reasoning inside the model. Keeping the steps apart means every decision decomposes into named
numbers.

**No privileged information at decision time.** The simulator *can* compute both outcomes
exactly — a real network cannot. So ground truth is manufactured offline, at training time
only. At decision time the agent sees nothing a deployed system could not observe.

**The model does not get the last word.** Deterministic gates can override it when buffers are
too low, when the options are too close to call, when the gain is too small to justify the
churn, or when the group switched too recently. A stall-escape path ensures a safety gate can
never trap a group in a mode that is actively failing. A separate policy layer then
re-validates the result independently, and can revert it.

**Provenance is explicit.** If the trained models are unavailable, the system computes outcomes
analytically instead of failing — and labels which path produced the result, so an output is
never presented as model-driven when it is not.

**Decisions are explained in plain language.** Each decision records a reason code that the API
renders as a readable sentence, so the dashboard can show the chosen mode, how decisively the
options separated, and why.

---

## Repository layout

| Path | Contents |
|---|---|
| `config/` | `SimConfig` — the single source of truth for every tunable parameter |
| `traffic_sim/simulation/` | Tick loop, cells, devices, mobility, buffers, traffic generation |
| `traffic_sim/network/` | Radio chain: path loss, received power, SINR, capacity, throughput |
| `traffic_sim/content/` | Synthetic content catalogue and subscription tracking |
| `traffic_sim/delivery/` | Unicast and broadcast delivery paths |
| `traffic_sim/ai/` | Features, training data, XGBoost models, utility, admission control, agent |
| `traffic_sim/mbos/` | Session and bearer coordination |
| `traffic_sim/policy/` | Policy enforcement layer |
| `traffic_sim/kpi/` | Metrics: continuity, QoE, buffer health, resource savings |
| `traffic_sim/experiments/` | Matched-seed policy comparison harness |
| `traffic_sim/api/` | FastAPI service and WebSocket tick stream |
| `frontend/` | React + TypeScript live dashboard |
| `tests/` | Radio math, delivery, AI layer, policy, and API coverage |

---

## Getting started

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run a simulation

```bash
# the AI decision layer
python -m traffic_sim.main --scenario flash_crowd --mode-selector ai

# fixed baselines, for comparison
python -m traffic_sim.main --scenario flash_crowd --mode-selector always-unicast
python -m traffic_sim.main --scenario flash_crowd --mode-selector always-broadcast
python -m traffic_sim.main --scenario flash_crowd --mode-selector rule-by-group-size
```

Results print as a summary and export to `data/generated/`.

### Compare policies on matched seeds

```bash
python -m traffic_sim.main --compare
```

Runs every policy against identical traffic, seeds, and content, so any difference is
attributable to the policy rather than to luck.

### Retrain the models

Trained models are committed, so the AI path works with no setup step.

```bash
python -m traffic_sim.ai.train
```

Retrain if you change the radio or capacity model — tree models cannot extrapolate beyond the
conditions they were trained on.

### Dashboard

```bash
pip install -e '.[api]'
python -m traffic_sim.api          # http://127.0.0.1:8000  (Swagger at /docs)
```

```bash
cd frontend
pnpm install
pnpm dev                           # http://127.0.0.1:5173
```

Start a run with a non-zero tick interval to watch decisions arrive live over the WebSocket.

### Tests

```bash
pip install -e '.[dev]'
pytest
```

---

## Scope and limitations

This is a research proof of concept, and the boundaries are deliberate:

- **The radio model is simplified.** Path loss, interference, SINR, and Shannon capacity are
  internally consistent and genuinely constrain delivery, but they are not a 3GPP or ATSC
  physical-layer implementation. ATSC 3.0 is the conceptual reference for the broadcast path;
  no broadcast hardware or protocol is involved.
- **Content is synthetic metadata.** No real video is encoded, segmented, or transmitted.
- **Coordinates are a synthetic 2D plane**, not real geography.
- **Multicast is out of scope**, as are deep-learning mobility prediction and reinforcement
  learning. XGBoost is the only model used.
- **At decision time the agent cannot know** how many devices will end up on unicast this tick,
  so it assumes the full cell population — the pessimistic case for unicast.
- **KPI instrumentation is being refined.** Some reported metrics are still being reworked to
  measure exactly what their names claim, so current dashboard figures are directional rather
  than final.

---

## License

**All rights reserved.** See [`LICENSE`](LICENSE).

This repository is public so it can be read and evaluated. It is **not** open source: no
permission is granted to use, copy, modify, or distribute the code. If you would like to use
any part of it, please get in touch first.
