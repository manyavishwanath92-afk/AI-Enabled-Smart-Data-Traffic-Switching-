# AI-Enabled Smart Data Traffic Switching

A software simulation / proof-of-concept of an AI-native telecom network that
dynamically switches content delivery between **unicast** and **broadcast**
at the content-segment level. This is a closed-loop simulation, not a real
5G core, RAN, or broadcast transmitter — see `docs/assumptions.md` style
caveats inline in each module's docstring.

## Project phases

This repository is built incrementally, per the project build guide:

- **Phase 1 (this phase)** — foundational network simulator: cells, UEs,
  mobility, synthetic content/subscriptions, radio math (path loss →
  received power → SINR → Shannon capacity), unicast/broadcast delivery,
  buffer dynamics, and a KPI engine. **No AI/ML, no utility model, no AI
  agent, no MBOS/PCF logic, and no dashboard yet** — delivery-mode
  selection is a pluggable, non-AI `mode_selector` (fixed rules only).
- **Phase 2+** — XGBoost prediction, utility model, AI agent (confidence,
  hysteresis, cooldown), MBOS/PCF, dashboard, always-unicast-vs-AI
  experiments. These live in already-created but currently empty stub
  packages: `traffic_sim/ai/`, `traffic_sim/policy/`, `traffic_sim/mbos/`,
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
- **Known limitation, inherited from the guide's own old-PoC constants:**
  the interference formula `I_linear = interference_coeff * active_users`
  (Guide Section 8.4, explicitly flagged there as "synthetic, not a
  physical interference model") is computed directly in linear
  milliwatts. At the guide's default transmit power / noise floor / path
  loss, this interference term dwarfs any realistic received signal power
  for any non-trivial active-user count, so `service_continuity`
  (SINR ≥ threshold) reads ~0 under default parameters regardless of
  delivery mode. The full radio chain (path loss → received power → SINR
  → Shannon capacity) is still computed and reported faithfully every
  tick as the guide requires (Steps 9-13), and its worked examples are
  unit-tested exactly; it is simply not used to gate actual buffer-fill
  throughput (see `traffic_sim/delivery/unicast.py`'s docstring), which
  is instead driven by the resource-accounting layer (Section 9) that the
  guide's own KPIs (bandwidth savings, congestion) are built around.
