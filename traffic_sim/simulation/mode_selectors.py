"""Fixed, non-AI mode_selector implementations for Phase 1 (Guide Step 20).

None of these are the AI agent — they are deliberately simple, fixed
rules used to (a) run always-unicast / always-broadcast baselines and
(b) exercise the Simulator's mode_selector seam that Phase 2's AI agent
will later occupy. `rule_by_group_size` in particular is explicitly a
placeholder threshold rule, not intelligence — the guide's core warning
("do not build a fixed rule such as 'users > X -> broadcast'") applies to
the *final production decision*, not to this Phase 1 scaffolding used
purely to prove the simulator's math produces sensible differences.
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


def rule_by_group_size(threshold: int = 20) -> ModeSelectorFn:
    def _selector(segment_id: str, ues: list[UE], cell: Cell, config: SimConfig) -> DeliveryMode:
        return DeliveryMode.BROADCAST if len(ues) >= threshold else DeliveryMode.UNICAST

    return _selector


SELECTOR_FACTORY = {
    "always-unicast": lambda rng: always_unicast,
    "always-broadcast": lambda rng: always_broadcast,
    "random": make_random_selector,
    "rule-by-group-size": lambda rng: rule_by_group_size(),
}
