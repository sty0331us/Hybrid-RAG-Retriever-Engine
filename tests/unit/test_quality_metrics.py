"""Unit tests for offline retrieval quality metrics."""

from __future__ import annotations

from hybrid_rag.core.types import RetrievalResult, RetrievedChunk
from hybrid_rag.evaluation.quality import (
    GoldenQuery,
    evaluate_golden_set,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    score_retrieval,
)


def test_precision_recall_mrr():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["c", "e"]
    assert precision_at_k(retrieved, relevant, k=3) == 1 / 3
    assert recall_at_k(retrieved, relevant, k=3) == 0.5
    assert mean_reciprocal_rank(retrieved, relevant) == 1 / 3


def test_score_retrieval_from_chunks():
    result = RetrievalResult(
        query="q",
        strategy="vector_store",
        vector_store="faiss",
        latency_ms=1.0,
        chunks=[
            RetrievedChunk(content="x", metadata={"source": "a.md"}, rank=1),
            RetrievedChunk(content="y", metadata={"source": "b.md"}, rank=2),
        ],
    )
    scores = score_retrieval(result, ["b.md"], k=2)
    assert scores.hit_at_k is True
    assert scores.mrr == 0.5
    assert scores.precision_at_k == 0.5


def test_evaluate_golden_set_aggregates():
    golden = GoldenQuery(query="q", relevant_doc_ids=["a.md"])
    result = RetrievalResult(
        query="q",
        strategy="ensemble",
        vector_store="faiss",
        latency_ms=1.0,
        chunks=[RetrievedChunk(content="x", metadata={"source": "a.md"}, rank=1)],
    )
    summary = evaluate_golden_set([(golden, result)], k=1)
    assert summary["count"] == 1
    assert summary["avg_mrr"] == 1.0
    assert summary["hit_rate"] == 1.0
