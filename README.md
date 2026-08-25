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
- **Phase 2b+** — XGBoost prediction, utility model, AI agent
  (confidence, hysteresis, cooldown), MBOS/PCF, dashboard,
  always-unicast-vs-AI experiments. These live in already-created but
  currently empty stub packages: `traffic_sim/ai/`,
  `traffic_sim/policy/`, `traffic_sim/mbos/`,
  `traffic_sim/visualization/`.

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
## Interference models (Phase 2)

The interference formula inherited from the old PoC,
`I_linear = interference_coeff * active_users` (Guide Section 8.4, flagged
there as "synthetic, not a physical interference model"), is
dimensionally unsound: it yields a bare milliwatt figure, while a
received signal at these Tx/path-loss defaults is ~6e-9 mW. At 30 active
users interference is 0.6 mW — roughly eight orders of magnitude above
the signal — so SINR collapsed to ~1e-9, Shannon capacity to ~0, and
`service_continuity` read 0.0000 in every Phase 1 run.

Phase 2 adds a calibrated **noise-rise** (interference-over-thermal)
model, selected by `interference_model` in `config/config.py`:

| model | formula | behaviour |
|---|---|---|
| `noise_rise` (default) | `coeff × active_users × noise_linear` | SINR ≈ 5–41 dB, falling with distance and cell load |
| `legacy_linear_mw` | `coeff × active_users` | Phase 1 baseline; degenerate, kept for comparison |

Expressing interference relative to the noise floor makes it
dimensionally consistent with the signal and noise terms it is summed
against. The legacy model is preserved and selectable — running with it
reproduces Phase 1's original numbers, including the fact that unicast
and broadcast become indistinguishable under it.

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
