"""Prometheus metrics unit tests."""

from __future__ import annotations

from hybrid_rag.observability.metrics import (
    metrics_payload,
    observe_ask,
    observe_retrieval,
)


def test_observe_retrieval_emits_prometheus_text():
    observe_retrieval(
        strategy="vector_store",
        vector_store="faiss",
        latency_ms=12.5,
        num_chunks=3,
    )
    payload = metrics_payload().decode("utf-8")
    assert "hybrid_rag_retrieval_latency_seconds" in payload
    assert "hybrid_rag_requests_total" in payload


def test_observe_ask_emits_generation_metric():
    observe_ask(
        strategy="ensemble",
        vector_store="chroma",
        model="gpt-4o-mini",
        retrieval_latency_ms=10.0,
        generation_latency_ms=100.0,
        num_chunks=2,
    )
    payload = metrics_payload().decode("utf-8")
    assert "hybrid_rag_generation_latency_seconds" in payload
    assert 'strategy="ensemble"' in payload
