"""Delivery-mode selectors that plug into the Simulator's `mode_selector` seam.

A selector answers one question per (cell, segment) group each tick:
UNICAST or BROADCAST? Everything here except the AI agent is a fixed,
non-intelligent rule, kept for baselines and comparison:

  always_unicast / always_broadcast  matched baselines (guide Section 22)
  make_random_selector               scrambles modes, used to generate
                                     diverse states for training data
  rule_by_group_size                 a deliberately naive size rule; the
                                     guide warns against shipping exactly
                                     this ("users > X -> broadcast"), so it
                                     exists only as a comparison point

The real decision-maker is `traffic_sim.ai.agent.AIAgent`, which is a
callable object matching `ModeSelectorFn` and is reached through the
"ai" factory entry below.
"""

from __future__ import annotations

import random
from typing import Callable

from config.config import SimConfig
from traffic_sim.simulation.cell import Cell
from traffic_sim.simulation.user import DeliveryMode, UE

ModeSelectorFn = Callable[[str, list[UE], Cell, SimConfig], DeliveryMode]


def always_unicast(segment_id: str, ues: list[UE], cell: Cell, config: SimConfig) -> DeliveryMode:
    return DeliveryMode.UNICAST


def always_broadcast(segment_id: str, ues: list[UE], cell: Cell, config: SimConfig) -> DeliveryMode:
    return DeliveryMode.BROADCAST


def make_random_selector(rng: random.Random) -> ModeSelectorFn:
    def _selector(segment_id: str, ues: list[UE], cell: Cell, config: SimConfig) -> DeliveryMode:
        return rng.choice([DeliveryMode.UNICAST, DeliveryMode.BROADCAST])

    return _selector


def rule_by_group_size(min_group_size: int = 20) -> ModeSelectorFn:
    """Broadcast once a group reaches `min_group_size`. Naive on purpose."""

    def _selector(segment_id: str, ues: list[UE], cell: Cell, config: SimConfig) -> DeliveryMode:
        return DeliveryMode.BROADCAST if len(ues) >= min_group_size else DeliveryMode.UNICAST

    return _selector


def _make_ai_selector(rng: random.Random, config: SimConfig) -> ModeSelectorFn:
    # Imported lazily so the simulator core never hard-depends on the AI
    # layer (or on xgboost being installed).
    from traffic_sim.ai.agent import AIAgent

    return AIAgent(config)


# Each entry takes (rng, config) so selectors that need configuration —
# the AI agent — can be built the same way as the trivial ones.
SELECTOR_FACTORY = {
    "always-unicast": lambda rng, config: always_unicast,
    "always-broadcast": lambda rng, config: always_broadcast,
    "random": lambda rng, config: make_random_selector(rng),
    "rule-by-group-size": lambda rng, config: rule_by_group_size(),
    "ai": _make_ai_selector,
}
