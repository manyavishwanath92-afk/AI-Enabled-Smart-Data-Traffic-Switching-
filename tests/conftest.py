import random

import pytest

from config.config import SimConfig


@pytest.fixture
def config() -> SimConfig:
    return SimConfig.default()


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)
