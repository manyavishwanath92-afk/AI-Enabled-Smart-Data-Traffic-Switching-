"""Deferred to Phase 2/3 per the project build guide.

Will eventually hold: controller.py (MBOS control-plane coordination of
the AI decision), session_manager.py, delivery_switch.py (data-plane
path/session coordination). This is a simulated controller only — not a
standards-compliant MBOS implementation. No logic exists here in Phase 1;
`traffic_sim.simulation.simulator.Simulator` invokes delivery directly via
its `mode_selector` seam.
"""
