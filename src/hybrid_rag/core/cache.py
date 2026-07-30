"""Simple in-memory TTL cache for retrieval and ask responses."""

from __future__ import annotations

import threading
import time
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """Thread-safe TTL cache with optional max size (FIFO eviction)."""

    def __init__(self, ttl_seconds: float, *, max_size: int = 256) -> None:
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.max_size = max(1, max_size)
        self._data: dict[Hashable, _Entry[T]] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.ttl_seconds > 0

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    def get(self, key: Hashable) -> T | None:
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at <= now:
                self._data.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return entry.value

    def set(self, key: Hashable, value: T) -> None:
        if not self.enabled:
            return
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            if key not in self._data and len(self._data) >= self.max_size:
                # Evict oldest insertion (dict preserves order on 3.7+).
                oldest = next(iter(self._data))
                self._data.pop(oldest, None)
            self._data[key] = _Entry(value=value, expires_at=expires_at)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "ttl_seconds": self.ttl_seconds,
                "size": len(self._data),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
            }
