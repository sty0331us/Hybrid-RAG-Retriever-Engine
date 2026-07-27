"""Vector store abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class BaseVectorStoreManager(ABC):
    """Lifecycle wrapper around a LangChain VectorStore backend."""

    backend_name: str

    def __init__(self, embeddings: Embeddings, collection_name: str) -> None:
        self.embeddings = embeddings
        self.collection_name = collection_name
        self._store: VectorStore | None = None

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            raise RuntimeError(
                f"{self.backend_name} store is not initialized. Call build() or load()."
            )
        return self._store

    @abstractmethod
    def build(self, documents: list[Document]) -> VectorStore:
        """Create an index from documents and persist it."""

    def create_empty(self) -> VectorStore:
        """Optional: create an empty index shell (override in backends that support it)."""
        raise NotImplementedError(f"{self.backend_name} does not support create_empty()")

    @abstractmethod
    def load(self) -> VectorStore:
        """Load a previously persisted index."""

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to an existing index."""

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        *,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Run similarity search."""

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        """Run similarity search returning relevance scores."""

    @abstractmethod
    def as_retriever(self, **kwargs: Any):
        """Return a LangChain Retriever bound to this store."""

    @abstractmethod
    def persist(self) -> None:
        """Flush index to disk if applicable."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Remove persisted artifacts for this collection."""
