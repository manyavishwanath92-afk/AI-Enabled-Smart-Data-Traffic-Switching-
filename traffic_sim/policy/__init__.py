"""Deferred to Phase 2+ per the project build guide.

Will eventually hold: pcf.py — the Policy Control Function module storing
thresholds, service priorities, confidence thresholds, hysteresis,
cooldown, and reporting-interval policy. No logic exists here in Phase 1;
those parameters are reserved (but unread) fields on
`config.config.SimConfig` for now.
"""
