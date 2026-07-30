"""REST API schemas for the Hybrid RAG Retriever Engine."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from hybrid_rag.config.settings import RetrieverStrategy, VectorStoreBackend


class HealthResponse(BaseModel):
    status: str = "ok"
    app_env: str
    default_retriever: str
    default_vector_store: str
    strategies: list[str]
    vector_stores: list[str]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    strategy: RetrieverStrategy = RetrieverStrategy.VECTOR_STORE
    backend: VectorStoreBackend = VectorStoreBackend.FAISS


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    strategy: RetrieverStrategy = RetrieverStrategy.VECTOR_STORE
    backend: VectorStoreBackend = VectorStoreBackend.FAISS


class ChunkResponse(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    rank: int = 0


class AskResponse(BaseModel):
    query: str
    answer: str
    strategy: str
    vector_store: str
    sources: list[ChunkResponse]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    model: str


class RetrieveResponse(BaseModel):
    query: str
    strategy: str
    vector_store: str
    chunks: list[ChunkResponse]
    latency_ms: float
    extras: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    backend: VectorStoreBackend = VectorStoreBackend.FAISS
    generate_answers: bool = True


class CompareResponse(BaseModel):
    summary: dict[str, Any]
    benchmarks: list[dict[str, Any]]


class IngestRequest(BaseModel):
    source: str = Field(..., description="File or directory path to ingest")
    backends: list[VectorStoreBackend] = Field(
        default_factory=lambda: [VectorStoreBackend.FAISS, VectorStoreBackend.CHROMA]
    )
    rebuild: bool = True


class IngestResponse(BaseModel):
    report: dict[str, Any]
