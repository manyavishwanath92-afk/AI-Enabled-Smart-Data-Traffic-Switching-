"""HTTP/WebSocket API layer over the simulator.

This package is a thin *serving* layer only. It builds simulators through the
existing `traffic_sim.main.build_simulator` seam, drives them one
`Simulator.step()` at a time, and serializes what they produce. It contains no
simulation, radio, delivery, AI, policy or KPI logic of its own — every number
it returns comes from the modules that already computed it.

    python -m traffic_sim.api          # start the server on 127.0.0.1:8000

See docs/FRONTEND_API_HANDOFF.md for the endpoint contract.
"""
