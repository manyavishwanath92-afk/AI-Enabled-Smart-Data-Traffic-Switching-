import random

import pytest

from config.config import SimConfig
from traffic_sim.content.content_server import ContentServer
from traffic_sim.content.subscription_manager import SubscriptionManager


def test_content_server_generates_expected_segment_count():
    cfg = SimConfig.default()
    server = ContentServer(cfg, random.Random(1))
    assert len(server.all_segments()) == cfg.num_segments


def test_content_server_initial_popularity_is_skewed_and_sums_to_one():
    cfg = SimConfig.default()
    server = ContentServer(cfg, random.Random(1))
    popularities = [s.popularity for s in server.all_segments()]
    assert sum(popularities) == pytest.approx(1.0, rel=1e-6)
    assert max(popularities) > min(popularities)  # skewed, not uniform


def test_popularity_formula_matches_guide():
    cfg = SimConfig.default()
    server = ContentServer(cfg, random.Random(1))
    manager = SubscriptionManager(server)

    segment_id = server.all_segments()[0].segment_id
    manager.record_request(segment_id, "ue_1")
    manager.record_request(segment_id, "ue_2")
    manager.record_request(segment_id, "ue_3")

    popularity = manager.compute_popularity(active_ue_count=10)
    assert popularity[segment_id] == 3 / 10
    assert server.get_segment(segment_id).popularity == 3 / 10


def test_clear_requests_resets_per_tick_demand():
    cfg = SimConfig.default()
    server = ContentServer(cfg, random.Random(1))
    manager = SubscriptionManager(server)
    segment_id = server.all_segments()[0].segment_id
    manager.record_request(segment_id, "ue_1")
    manager.clear_requests()
    assert manager.requesting_ues_for_segment(segment_id) == []


def test_requesting_ues_never_exceed_active_ues_with_one_request_per_ue():
    cfg = SimConfig.default()
    server = ContentServer(cfg, random.Random(1))
    manager = SubscriptionManager(server)
    ue_ids = [f"ue_{i}" for i in range(20)]
    rng = random.Random(2)
    for ue_id in ue_ids:
        segment_id = rng.choice(server.all_segments()).segment_id
        manager.record_request(segment_id, ue_id)

    total_requests = sum(len(v) for v in manager.all_requests().values())
    assert total_requests == len(ue_ids)
