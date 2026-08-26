"""Deferred to Phase 2+ per the project build guide.

Will eventually hold: pcf.py — the Policy Control Function module storing
service priorities, minimum-confidence limits, hysteresis, cooldown, and
reporting-interval policy as a first-class module. For now those
parameters live directly on `config.config.SimConfig` and are read by
`traffic_sim.ai.agent`.
"""
