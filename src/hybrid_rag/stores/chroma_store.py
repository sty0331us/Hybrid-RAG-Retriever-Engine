"""ChromaDB vector store manager."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from hybrid_rag.core.exceptions import StoreError
from hybrid_rag.core.logging import get_logger
from hybrid_rag.stores.base import BaseVectorStoreManager

logger = get_logger(__name__)


class ChromaStoreManager(BaseVectorStoreManager):
    backend_name = "chroma"

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        persist_dir: Path,
        collection_name: str = "hybrid_rag_docs",
    ) -> None:
        super().__init__(embeddings, collection_name)
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def _client_kwargs(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "embedding_function": self.embeddings,
            "persist_directory": str(self.persist_dir),
        }

    def create_empty(self) -> VectorStore:
        try:
            self.delete_collection()
            self._store = Chroma(**self._client_kwargs())
            logger.info("chroma_empty_created", collection=self.collection_name)
            return self._store
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"Chroma empty init failed: {exc}") from exc

    def build(self, documents: list[Document]) -> VectorStore:
        if not documents:
            return self.create_empty()
        try:
            # Fresh collection on rebuild
            self.delete_collection()
            self._store = Chroma.from_documents(documents, **self._client_kwargs())
            logger.info(
                "chroma_built",
                documents=len(documents),
                collection=self.collection_name,
            )
            return self._store
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"Chroma build failed: {exc}") from exc

    def load(self) -> VectorStore:
        try:
            self._store = Chroma(**self._client_kwargs())
            # Touch collection to validate
            _ = self._store._collection.count()  # noqa: SLF001
            logger.info("chroma_loaded", collection=self.collection_name)
            return self._store
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"Chroma load failed: {exc}") from exc

    def add_documents(self, documents: list[Document]) -> list[str]:
        return self.store.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        *,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        if filter:
            return self.store.similarity_search(query, k=k, filter=filter)
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k)

    def as_retriever(self, **kwargs: Any):
        return self.store.as_retriever(**kwargs)

    def persist(self) -> None:
        # Chroma with persist_directory auto-persists
        logger.debug("chroma_persist_noop", collection=self.collection_name)

    def delete_collection(self) -> None:
        # Best-effort remove collection files; Chroma stores under persist_dir
        self._store = None
        # Do not wipe entire persist_dir if multiple collections share it —
        # recreate empty client and delete named collection if possible.
        try:
            client = Chroma(**self._client_kwargs())
            client.delete_collection()
        except Exception:  # noqa: BLE001
            # Fallback: wipe persist dir when only one collection is expected
            if self.persist_dir.exists() and any(self.persist_dir.iterdir()):
                shutil.rmtree(self.persist_dir)
                self.persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info("chroma_deleted", collection=self.collection_name)
