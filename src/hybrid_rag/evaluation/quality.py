"""Offline retrieval quality metrics (precision@k, recall@k, MRR)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from hybrid_rag.core.types import RetrievalResult, RetrievedChunk


class GoldenQuery(BaseModel):
    """A labeled query used for offline retrieval evaluation."""

    query: str
    relevant_doc_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class QualityScores(BaseModel):
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_at_k: bool
    k: int
    retrieved_ids: list[str] = Field(default_factory=list)
    relevant_ids: list[str] = Field(default_factory=list)


def chunk_doc_id(chunk: RetrievedChunk) -> str | None:
    meta = chunk.metadata or {}
    for key in ("doc_id", "source", "title", "id"):
        value = meta.get(key)
        if value:
            return str(value)
    return None


def retrieved_ids(chunks: Sequence[RetrievedChunk], *, k: int | None = None) -> list[str]:
    selected = chunks if k is None else chunks[:k]
    ids: list[str] = []
    for chunk in selected:
        doc_id = chunk_doc_id(chunk)
        if doc_id is not None:
            ids.append(doc_id)
    return ids


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], *, k: int) -> float:
    if k <= 0:
        return 0.0
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in top if doc_id in relevant_set)
    return hits / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], *, k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = list(retrieved)[:k]
    hits = sum(1 for doc_id in top if doc_id in relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def score_retrieval(
    result: RetrievalResult,
    relevant_doc_ids: Sequence[str],
    *,
    k: int = 4,
) -> QualityScores:
    ids = retrieved_ids(result.chunks, k=k)
    precision = precision_at_k(ids, relevant_doc_ids, k=k)
    recall = recall_at_k(ids, relevant_doc_ids, k=k)
    mrr = mean_reciprocal_rank(ids, relevant_doc_ids)
    return QualityScores(
        precision_at_k=round(precision, 4),
        recall_at_k=round(recall, 4),
        mrr=round(mrr, 4),
        hit_at_k=any(doc_id in set(relevant_doc_ids) for doc_id in ids),
        k=k,
        retrieved_ids=ids,
        relevant_ids=list(relevant_doc_ids),
    )


def evaluate_golden_set(
    results: Sequence[tuple[GoldenQuery, RetrievalResult]],
    *,
    k: int = 4,
) -> dict[str, Any]:
    """Aggregate quality metrics across a golden query set."""
    if not results:
        return {"count": 0, "k": k}
    scores = [score_retrieval(result, golden.relevant_doc_ids, k=k) for golden, result in results]
    return {
        "count": len(scores),
        "k": k,
        "avg_precision_at_k": round(sum(s.precision_at_k for s in scores) / len(scores), 4),
        "avg_recall_at_k": round(sum(s.recall_at_k for s in scores) / len(scores), 4),
        "avg_mrr": round(sum(s.mrr for s in scores) / len(scores), 4),
        "hit_rate": round(sum(1 for s in scores if s.hit_at_k) / len(scores), 4),
        "per_query": [
            {
                "query": golden.query,
                **score.model_dump(),
            }
            for (golden, _), score in zip(results, scores, strict=True)
        ],
    }
