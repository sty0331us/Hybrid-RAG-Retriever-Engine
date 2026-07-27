"""FAISS vector store manager."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from hybrid_rag.core.exceptions import StoreError
from hybrid_rag.core.logging import get_logger
from hybrid_rag.stores.base import BaseVectorStoreManager

logger = get_logger(__name__)


class FaissStoreManager(BaseVectorStoreManager):
    backend_name = "faiss"

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        index_dir: Path,
        collection_name: str = "hybrid_rag_docs",
    ) -> None:
        super().__init__(embeddings, collection_name)
        self.index_dir = index_dir / collection_name
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def create_empty(self) -> VectorStore:
        """Initialize an empty FAISS index (needed by ParentDocumentRetriever)."""
        try:
            dim = len(self.embeddings.embed_query("__dim_probe__"))
            index = faiss.IndexFlatL2(dim)
            self._store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            self.persist()
            logger.info("faiss_empty_created", dim=dim, path=str(self.index_dir))
            return self._store
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"FAISS empty init failed: {exc}") from exc

    def build(self, documents: list[Document]) -> VectorStore:
        if not documents:
            return self.create_empty()
        try:
            self._store = FAISS.from_documents(documents, self.embeddings)
            self.persist()
            logger.info("faiss_built", documents=len(documents), path=str(self.index_dir))
            return self._store
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"FAISS build failed: {exc}") from exc

    def load(self) -> VectorStore:
        index_file = self.index_dir / "index.faiss"
        if not index_file.exists():
            raise StoreError(f"FAISS index not found at {self.index_dir}")
        try:
            self._store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("faiss_loaded", path=str(self.index_dir))
            return self._store
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"FAISS load failed: {exc}") from exc

    def add_documents(self, documents: list[Document]) -> list[str]:
        ids = self.store.add_documents(documents)
        self.persist()
        return ids

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
        if self._store is None:
            return
        self._store.save_local(str(self.index_dir))
        logger.debug("faiss_persisted", path=str(self.index_dir))

    def delete_collection(self) -> None:
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)
        self._store = None
        logger.info("faiss_deleted", path=str(self.index_dir))
