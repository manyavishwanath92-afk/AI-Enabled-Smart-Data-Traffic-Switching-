# AI-Enabled Smart Data Traffic Switching

A software simulation / proof-of-concept of an AI-native telecom network that
dynamically switches content delivery between **unicast** and **broadcast**
at the content-segment level. This is a closed-loop simulation, not a real
5G core, RAN, or broadcast transmitter — see `docs/assumptions.md` style
caveats inline in each module's docstring.

## Project phases

This repository is built incrementally, per the project build guide:

- **Phase 1 (done)** — foundational network simulator: cells, UEs,
  mobility, synthetic content/subscriptions, radio math (path loss →
  received power → SINR → Shannon capacity), unicast/broadcast delivery,
  buffer dynamics, and a KPI engine. Delivery-mode selection is a
  pluggable, non-AI `mode_selector` (fixed rules only).
- **Phase 2a (this phase)** — make the radio model actually drive
  delivery: a calibrated interference model, an achievable-rate layer
  linking Shannon capacity to the buffer, coherent segment bitrates, and
  playback-continuity KPIs. Still **no AI/ML, no utility model, no AI
  agent, no MBOS/PCF logic, and no dashboard.**
- **Phase 2b/3 (this phase)** — the intelligence layer: features →
  XGBoost outcome prediction → utility → AI agent → delivery →
  closed-loop feedback, plus a matched comparison harness. Lives in
  `traffic_sim/ai/` and `traffic_sim/experiments/`.
- **Later** — MBOS/PCF as first-class modules and a dashboard. Still
  empty stubs: `traffic_sim/policy/`, `traffic_sim/mbos/`,
  `traffic_sim/visualization/`.

## The intelligence layer

```
network state → features → XGBoost (predicts each mode's outcome)
    → utility (scores the trade-off) → agent (applies policy & safety)
    → UNICAST / BROADCAST → simulator → changed state → feedback
```

The three steps are deliberately separate and individually inspectable:

| Step | File | Answers |
|---|---|---|
| Features | `ai/feature_engineering.py` | what do we know right now? |
| Prediction | `ai/predict.py`, `ai/train.py` | what will each mode do? |
| Utility | `ai/utility.py` | which outcome do we prefer, and why? |
| Agent | `ai/agent.py` | what are we allowed to do about it? |

**Why ML when the simulator could just compute the answer?** A real
network cannot try both modes and keep the better one. The simulator can,
so it manufactures ground truth *at training time only*
(`ai/dataset.py`). At decision time the agent sees telemetry alone. If
the model is missing the agent falls back to computing outcomes
analytically and says so via `PredictionResult.source`, so results are
never mislabelled as model-driven.

XGBoost is the only ML model used. Two regressors (one per mode) predict
achievable delivery rate; R² ≈ 0.999, MAE ≈ 0.01 Mbps.

```bash
python -m traffic_sim.ai.train                      # retrain (models are committed)
python -m traffic_sim.main --mode-selector ai       # run the agent
python -m traffic_sim.main --compare                # matched policy comparison
```

### Measured results

Matched seeds, flash-crowd, 60 ticks. **Default carrier (60 Mbps)** —
broadcast is genuinely the right answer, and the AI finds it:

| policy | resource saved | continuity | QoE | switches |
|---|---|---|---|---|
| always-unicast | 0.000 | 0.713 | 0.201 | 0 |
| always-broadcast | 0.844 | 1.000 | 0.486 | 0 |
| rule-by-group-size | 0.293 | 0.820 | 0.264 | 0 |
| **ai** | **0.821** | **1.000** | **0.490** | 4 |

**Constrained carrier (4 Mbps)** — broadcasting everything now destroys
service, and the AI backs off:

| policy | resource saved | continuity | QoE | switches |
|---|---|---|---|---|
| always-unicast | 0.000 | 0.755 | 0.224 | 0 |
| always-broadcast | 0.841 | **0.457** | 0.075 | 0 |
| rule-by-group-size | 0.302 | **0.864** | 0.294 | 0 |
| **ai** | 0.267 | 0.840 | 0.263 | 98.7 |

**Honest reading.** The AI adapts — broadcast share falls from ~97% to
~27% as the carrier shrinks, and it beats always-broadcast on continuity
by a wide margin (0.840 vs 0.457) where blind broadcasting collapses.
But in this constrained regime the naive group-size rule still edges it
out on both continuity (0.864) and savings (0.302), and the AI churns
badly (98.7 switches vs 0).

The cause is a coordination problem, not a tuning problem: every group
reads the same shared-carrier signal and reacts identically, so they
leave together, the carrier looks free, and they pile back. Exponentially
smoothing that signal was tried — churn fell to 79 switches but
continuity and savings got slightly worse, so it was not kept. Fixing
this properly needs cell-level admission control (deciding how many
streams the carrier admits) rather than independent per-group decisions.

Two known limitations worth stating plainly:
- Tree models cannot extrapolate. An early version trained at a single
  broadcast capacity learned to ignore that feature and kept choosing
  broadcast into a starved carrier. `ai/dataset.py` now sweeps capacity
  settings; `broadcast_capacity_mbps` became the top broadcast feature
  (0.563 importance). Retrain if you change the physics.
- At decision time the agent cannot know how many UEs will end up on
  unicast this tick, so it assumes the whole cell population — the
  pessimistic case for unicast.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

## Running the simulator

```bash
python -m traffic_sim.main --scenario flash_crowd --mode-selector always-broadcast
python -m traffic_sim.main --scenario flash_crowd --mode-selector always-unicast
```

Output is printed as a summary table and exported to `data/generated/`.

## Running tests

```bash
pytest
```

## Key simplifying assumptions (Phase 1)

- Radio equations (path loss, interference, SINR, Shannon capacity) are
  deliberately simplified PoC assumptions, not a 3GPP/ATSC PHY-layer
  implementation. See `traffic_sim/network/` module docstrings.
- Content is synthetic metadata only; no real video/segmentation.
- Coordinates are a synthetic 2D Cartesian plane (km), not real geography.
- Multicast is out of scope entirely (deferred indefinitely per the source
  use case). Deep-learning mobility prediction, reinforcement learning, and
  real operator-network integration are also out of scope.
## Interference model

Interference uses a calibrated **noise-rise** (interference-over-thermal)
model, in `config/config.py` / `traffic_sim/network/sinr.py`:

```
I_linear = interference_rise_coeff * active_users * noise_linear
```

Interference is expressed *relative to the noise floor* rather than as
an absolute milliwatt figure, which keeps it dimensionally consistent
with the signal and noise terms it is summed against. At the defaults
this yields SINR spanning roughly 5–41 dB, falling with both distance
and cell load.

An earlier formula inherited from an older PoC,
`I_linear = coeff * active_users` (no noise term), was dimensionally
unsound: it produced a bare milliwatt figure, while a received signal at
these Tx/path-loss defaults is ~6e-9 mW. At 30 active users that
interference term was 0.6 mW — roughly eight orders of magnitude above
the signal — so SINR collapsed to ~1e-9 regardless of distance or load,
Shannon capacity collapsed to ~0, and the delivery chain could not move
any content at all. It has been removed rather than kept as a selectable
option: this simulator implements one interference model.

## Service continuity

Service continuity has exactly one definition, computed from actual
playback stalls in `traffic_sim/kpi/reliability.py`:

```
SC = 1 - (total playback interruption time / total playback time)
```

SINR does **not** define continuity directly. It reaches
continuity only through the delivery chain — SINR sets Shannon capacity,
which sets the achievable delivery rate, which is raced against the
segment's bitrate to fill or drain the UE buffer; an empty buffer is a
stall, and stalls are what SC counts. Raw radio quality is reported
separately as the `mean_sinr_db` KPI.

**The radio chain now gates delivery.** In Phase 1 SINR and Shannon
capacity were computed, reported, and then discarded; throughput was
decided purely by resource accounting. The full chain is now closed:

```
distance → path loss → received power → SINR → Shannon capacity
        → achievable delivery rate → content received → buffer → playback continuity
```

Unicast UEs share both the radio-limited and resource-limited capacity
with the other UEs their cell is serving by unicast, so throughput
degrades as a cell fills. A broadcast transmission is sized at the
group's worst-case SINR and is *not* divided by group size, which is the
physical reason it wins under a flash crowd. Because delivery modes are
decided for all groups before any are delivered, offloading a group to
broadcast genuinely frees cellular capacity for the UEs that remain on
unicast.
