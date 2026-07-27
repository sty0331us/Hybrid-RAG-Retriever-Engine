"""Unit tests for chunking and metrics (no API key required)."""

from __future__ import annotations

from langchain_core.documents import Document

from hybrid_rag.config.settings import Settings
from hybrid_rag.core.types import RetrievalResult, RetrievedChunk
from hybrid_rag.evaluation.metrics import (
    average_score,
    summarize_comparison,
    to_benchmark,
    unique_source_count,
)
from hybrid_rag.ingestion.chunker import chunk_documents, chunk_parent_child


def _settings(**overrides) -> Settings:
    base = {
        "openai_api_key": "sk-test",
        "chunk_size": 200,
        "chunk_overlap": 40,
        "parent_chunk_size": 400,
        "child_chunk_size": 100,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_flat_chunking_assigns_indices():
    docs = [
        Document(
            page_content="A" * 500,
            metadata={"source": "a.md", "doc_id": "abc", "category": "rag"},
        )
    ]
    chunks = chunk_documents(docs, _settings(), mode="flat")
    assert len(chunks) > 1
    assert all("chunk_index" in c.metadata for c in chunks)


def test_parent_child_linkage():
    docs = [
        Document(
            page_content=("Parent body. " * 80),
            metadata={"source": "p.md", "category": "rag", "title": "Parent"},
        )
    ]
    parents, children = chunk_parent_child(docs, _settings())
    assert parents
    assert children
    assert all(c.metadata.get("parent_id") for c in children)
    parent_ids = {p.metadata["doc_id"] for p in parents}
    assert {c.metadata["parent_id"] for c in children} <= parent_ids


def test_metrics_benchmark_summary():
    result = RetrievalResult(
        query="q",
        strategy="vector_store",
        vector_store="faiss",
        latency_ms=12.5,
        chunks=[
            RetrievedChunk(content="a", metadata={"source": "1"}, score=0.9, rank=1),
            RetrievedChunk(content="b", metadata={"source": "2"}, score=0.7, rank=2),
        ],
    )
    assert average_score(result) == 0.8
    assert unique_source_count(result) == 2
    bench = to_benchmark(result)
    summary = summarize_comparison([bench])
    assert summary["fastest_strategy"] == "vector_store"
    assert summary["count"] == 1
