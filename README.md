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
| always-unicast | 0.000 | 0.678 | 0.186 | 0 |
| always-broadcast | 0.839 | 1.000 | 0.485 | 0 |
| rule-by-group-size | 0.289 | 0.795 | 0.250 | 0 |
| **ai** | 0.758 | **1.000** | **0.497** | 50.7 |

Switch count is higher here than the constrained-carrier case below —
see the note under "Cell-level coordination" for why: with an abundant
carrier and scattered background traffic, a cell can have 6-8 concurrent
broadcast candidates, and which exact set fits shifts slightly as
individual UEs' radio conditions drift tick to tick. Outcomes (savings,
continuity, QoE) are barely affected; this is a churn cost that was not
present before coordination was added and is flagged here rather than
hidden, in the same spirit as the constrained-carrier tradeoffs below.

**Constrained carrier (4 Mbps)** — broadcasting everything now destroys
service, and the AI backs off:

| policy | resource saved | continuity | QoE | switches |
|---|---|---|---|---|
| always-unicast | 0.000 | 0.713 | 0.202 | 0 |
| always-broadcast | 0.846 | **0.442** | 0.072 | 0 |
| rule-by-group-size | 0.302 | 0.829 | 0.270 | 0 |
| **ai** | 0.186 | **0.865** | **0.303** | 37.7 |

(Same matched seeds and tick count as the default-carrier table above;
figures are the `--compare` output for `broadcast_cell_capacity_mbps=4.0`.)

**Cell-level coordination.** Independently-decided groups produced the
same herd oscillation described above — every group reacted to the same
shared-carrier contention reading, so a busy carrier made the whole cell
leave broadcast at once, which made it look free, which drew them all
back (98.7 switches/60 ticks). `traffic_sim/ai/cell_controller.py` fixes
this by deciding once per cell instead of once per group: every
candidate is ranked by `broadcast_benefit = max(0, U_broadcast −
U_unicast)`, then admitted to the shared carrier highest-benefit-first,
only while the carrier can still sustain the bitrate of every group
already admitted (both its raw `broadcast_capacity_mbps` and each
group's own worst-case Shannon capacity, divided across everyone
sharing it). Groups that don't fit stay on unicast.

This took switching **98.7 → 37.7** (62% fewer) and flipped the
comparison against the naive rule: before coordination the AI's
continuity (0.840) trailed the rule's (0.864); afterward the AI's (0.865)
**beats** it. Diagnosing the remaining churn found the ranking itself was
unstable at the margin — a group's own utility gap stays large even when
it's the admission *boundary* that's wobbling, so the existing hysteresis
gate (which only looks at raw utility gap) never damped it. The fix
reuses `hysteresis_margin` as a small ranking tie-break favouring
whichever group already holds a broadcast seat, so a newcomer must be
genuinely better, not just marginally luckier this tick, to bump it.

Two prior, smaller-scope attempts were tried and rejected because they
didn't clearly pay off: exponentially smoothing the shared contention
signal cut churn to 79 but slightly hurt continuity and savings; a
looser admission check helped less than the ranking fix. Both were
reverted rather than kept as unproven complexity.

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
