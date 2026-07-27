"""Retrieval quality & latency metrics."""

from __future__ import annotations

from hybrid_rag.core.types import RAGAnswer, RetrievalResult, StrategyBenchmark


def average_score(result: RetrievalResult) -> float | None:
    scores = [c.score for c in result.chunks if c.score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def unique_source_count(result: RetrievalResult) -> int:
    sources = {
        (c.metadata.get("source") or c.metadata.get("title") or c.content[:40])
        for c in result.chunks
    }
    return len(sources)


def to_benchmark(
    result: RetrievalResult | RAGAnswer,
    *,
    notes: str = "",
) -> StrategyBenchmark:
    if isinstance(result, RAGAnswer):
        retrieval_like = RetrievalResult(
            query=result.query,
            strategy=result.strategy,
            vector_store=result.vector_store,
            chunks=result.sources,
            latency_ms=result.retrieval_latency_ms,
        )
        preview = result.answer[:240]
        latency = result.total_latency_ms
    else:
        retrieval_like = result
        preview = ""
        latency = result.latency_ms
        if result.extras.get("error"):
            notes = notes or str(result.extras["error"])

    return StrategyBenchmark(
        strategy=retrieval_like.strategy,
        vector_store=retrieval_like.vector_store,
        query=retrieval_like.query,
        latency_ms=round(latency, 2),
        num_chunks=len(retrieval_like.chunks),
        avg_score=(
            round(average_score(retrieval_like), 4)
            if average_score(retrieval_like) is not None
            else None
        ),
        unique_sources=unique_source_count(retrieval_like),
        answer_preview=preview,
        notes=notes,
    )


def rank_by_latency(benchmarks: list[StrategyBenchmark]) -> list[StrategyBenchmark]:
    return sorted(benchmarks, key=lambda b: b.latency_ms)


def summarize_comparison(benchmarks: list[StrategyBenchmark]) -> dict:
    if not benchmarks:
        return {"count": 0}
    fastest = min(benchmarks, key=lambda b: b.latency_ms)
    most_diverse = max(benchmarks, key=lambda b: b.unique_sources)
    return {
        "count": len(benchmarks),
        "fastest_strategy": fastest.strategy,
        "fastest_latency_ms": fastest.latency_ms,
        "most_diverse_strategy": most_diverse.strategy,
        "most_diverse_sources": most_diverse.unique_sources,
        "strategies": [b.model_dump() for b in benchmarks],
    }
