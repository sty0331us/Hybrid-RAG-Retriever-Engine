"""TTL cache unit tests."""

from __future__ import annotations

import time

from hybrid_rag.core.cache import TTLCache


def test_ttl_cache_hit_and_miss():
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)
    assert cache.get("a") is None
    cache.set("a", "value")
    assert cache.get("a") == "value"
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_ttl_cache_expiry():
    cache: TTLCache[int] = TTLCache(ttl_seconds=0.05)
    cache.set("x", 1)
    assert cache.get("x") == 1
    time.sleep(0.06)
    assert cache.get("x") is None


def test_ttl_disabled_when_zero():
    cache: TTLCache[str] = TTLCache(ttl_seconds=0)
    cache.set("a", "value")
    assert cache.get("a") is None
    assert cache.enabled is False


def test_ttl_evicts_oldest_when_full():
    cache: TTLCache[str] = TTLCache(ttl_seconds=60, max_size=2)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.set("c", "3")
    assert cache.get("a") is None
    assert cache.get("b") == "2"
    assert cache.get("c") == "3"
