"""Deferred to Phase 2+ per the project build guide.

Will eventually hold: feature_engineering.py, dataset.py, train.py,
predict.py (XGBoost prediction), utility.py (multi-objective utility
model), agent.py (AI agent: confidence, safety checks, hysteresis,
cooldown). No logic exists here in Phase 1 — delivery-mode selection in
Phase 1 is a fixed, non-AI `mode_selector` callable passed into
`traffic_sim.simulation.simulator.Simulator`, not this package.

Explicitly out of scope even in later phases: multicast, reinforcement
learning, deep-learning mobility prediction, real operator-network
integration.
"""
