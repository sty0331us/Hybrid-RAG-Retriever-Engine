"""Shared domain types used across packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentMeta(BaseModel):
    """Normalized metadata attached to every chunk."""

    source: str
    doc_id: str
    category: str = "general"
    title: str = ""
    year: int | None = None
    page: int | None = None
    chunk_index: int = 0
    parent_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    rank: int = 0


class RetrievalResult(BaseModel):
    query: str
    strategy: str
    vector_store: str
    chunks: list[RetrievedChunk]
    latency_ms: float
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    extras: dict[str, Any] = Field(default_factory=dict)


class RAGAnswer(BaseModel):
    query: str
    answer: str
    strategy: str
    vector_store: str
    sources: list[RetrievedChunk]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    model: str


class StrategyBenchmark(BaseModel):
    strategy: str
    vector_store: str
    query: str
    latency_ms: float
    num_chunks: int
    avg_score: float | None = None
    unique_sources: int = 0
    answer_preview: str = ""
    notes: str = ""


@dataclass
class IngestStats:
    files_processed: int = 0
    documents_loaded: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
