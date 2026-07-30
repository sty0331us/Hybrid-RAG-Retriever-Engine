"""Prometheus metrics for retrieval and generation latency."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, generate_latest

from hybrid_rag.core.logging import get_logger

logger = get_logger(__name__)

RETRIEVAL_LATENCY = Histogram(
    "hybrid_rag_retrieval_latency_seconds",
    "Retrieval latency in seconds",
    ["strategy", "vector_store"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

GENERATION_LATENCY = Histogram(
    "hybrid_rag_generation_latency_seconds",
    "LLM generation latency in seconds",
    ["strategy", "vector_store", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

REQUESTS_TOTAL = Counter(
    "hybrid_rag_requests_total",
    "Total RAG pipeline requests",
    ["operation", "strategy", "vector_store", "status"],
)

CHUNKS_RETRIEVED = Histogram(
    "hybrid_rag_chunks_retrieved",
    "Number of chunks returned per retrieval",
    ["strategy", "vector_store"],
    buckets=(0, 1, 2, 4, 8, 16, 32),
)


def observe_retrieval(
    *,
    strategy: str,
    vector_store: str,
    latency_ms: float,
    num_chunks: int,
    status: str = "ok",
) -> None:
    labels = {"strategy": strategy, "vector_store": vector_store}
    RETRIEVAL_LATENCY.labels(**labels).observe(latency_ms / 1000.0)
    CHUNKS_RETRIEVED.labels(**labels).observe(num_chunks)
    REQUESTS_TOTAL.labels(
        operation="retrieve",
        strategy=strategy,
        vector_store=vector_store,
        status=status,
    ).inc()


def observe_ask(
    *,
    strategy: str,
    vector_store: str,
    model: str,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    num_chunks: int,
    status: str = "ok",
) -> None:
    observe_retrieval(
        strategy=strategy,
        vector_store=vector_store,
        latency_ms=retrieval_latency_ms,
        num_chunks=num_chunks,
        status=status,
    )
    GENERATION_LATENCY.labels(
        strategy=strategy,
        vector_store=vector_store,
        model=model,
    ).observe(generation_latency_ms / 1000.0)
    REQUESTS_TOTAL.labels(
        operation="ask",
        strategy=strategy,
        vector_store=vector_store,
        status=status,
    ).inc()


def metrics_payload() -> bytes:
    """Render Prometheus text exposition format."""
    return generate_latest()
