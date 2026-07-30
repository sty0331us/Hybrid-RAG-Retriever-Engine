"""Unit tests for ensemble hybrid retrieval (dense + lexical RRF)."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from hybrid_rag.retrievers.ensemble import (
    EnsembleHybridRetrieverStrategy,
    lexical_score,
    reciprocal_rank_fusion,
    tokenize,
)
from hybrid_rag.stores.base import BaseVectorStoreManager


class _FakeStore(BaseVectorStoreManager):
    backend_name = "fake"

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs
        self.embeddings = object()  # type: ignore[assignment]
        self.collection_name = "fake"
        self._store = object()  # type: ignore[assignment]

    def build(self, documents: list[Document]) -> VectorStore:
        raise NotImplementedError

    def load(self) -> VectorStore:
        raise NotImplementedError

    def add_documents(self, documents: list[Document]) -> list[str]:
        raise NotImplementedError

    def similarity_search(self, query: str, *, k: int = 4, filter=None) -> list[Document]:
        return self._docs[:k]

    def similarity_search_with_score(self, query: str, *, k: int = 4):
        return [(d, 1.0) for d in self._docs[:k]]

    def as_retriever(self, **kwargs):
        raise NotImplementedError

    def persist(self) -> None:
        return None

    def delete_collection(self) -> None:
        return None


def test_tokenize_and_lexical_score():
    assert "faiss" in tokenize("FAISS vs Chroma")
    assert lexical_score("FAISS index", "Building a FAISS index for vectors") > lexical_score(
        "FAISS index", "Completely unrelated gardening tips"
    )


def test_reciprocal_rank_fusion_prefers_consensus():
    fused = reciprocal_rank_fusion(
        [
            ["a", "b", "c"],
            ["b", "a", "d"],
        ],
        k=60,
    )
    assert fused[0][0] in {"a", "b"}
    assert {doc_id for doc_id, _ in fused} == {"a", "b", "c", "d"}


def test_ensemble_retrieve_fuses_rankings():
    docs = [
        Document(
            page_content="Semantic similarity search with embeddings",
            metadata={"doc_id": "d1", "source": "sem.md", "chunk_index": 0},
        ),
        Document(
            page_content="FAISS is a library for efficient similarity search",
            metadata={"doc_id": "d2", "source": "faiss.md", "chunk_index": 0},
        ),
        Document(
            page_content="Gardening tips for tomatoes",
            metadata={"doc_id": "d3", "source": "garden.md", "chunk_index": 0},
        ),
    ]
    strategy = EnsembleHybridRetrieverStrategy(_FakeStore(docs), top_k=2, fetch_k=3)
    result = strategy.retrieve("FAISS similarity search")
    assert result.strategy == "ensemble"
    assert len(result.chunks) == 2
    assert result.extras["fusion"] == "rrf"
    # Keyword-heavy FAISS doc should surface in hybrid top results.
    contents = " ".join(c.content for c in result.chunks)
    assert "FAISS" in contents
