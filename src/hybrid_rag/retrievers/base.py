"""Retriever strategy abstractions."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from hybrid_rag.core.types import RetrievalResult, RetrievedChunk


class BaseStrategyRetriever(ABC):
    """Uniform interface over LangChain retriever strategies."""

    strategy_name: str

    def __init__(self, vector_store_name: str, top_k: int = 4) -> None:
        self.vector_store_name = vector_store_name
        self.top_k = top_k
        self._retriever: BaseRetriever | None = None

    @abstractmethod
    def build(self) -> BaseRetriever:
        """Construct the underlying LangChain retriever."""

    @property
    def retriever(self) -> BaseRetriever:
        if self._retriever is None:
            self._retriever = self.build()
        return self._retriever

    def retrieve(self, query: str, **kwargs: Any) -> RetrievalResult:
        started = time.perf_counter()
        docs: list[Document] = self.retriever.invoke(query, **kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        chunks = [
            RetrievedChunk(
                content=doc.page_content,
                metadata=dict(doc.metadata),
                score=_extract_score(doc),
                rank=i + 1,
            )
            for i, doc in enumerate(docs)
        ]
        return RetrievalResult(
            query=query,
            strategy=self.strategy_name,
            vector_store=self.vector_store_name,
            chunks=chunks,
            latency_ms=latency_ms,
        )


def _extract_score(doc: Document) -> float | None:
    for key in ("score", "relevance_score", "_distance"):
        if key in doc.metadata and doc.metadata[key] is not None:
            try:
                return float(doc.metadata[key])
            except (TypeError, ValueError):
                return None
    return None
