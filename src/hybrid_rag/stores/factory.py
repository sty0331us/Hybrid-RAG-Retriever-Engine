"""Factory for vector store backends."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from hybrid_rag.config.settings import Settings, VectorStoreBackend
from hybrid_rag.core.exceptions import ConfigurationError
from hybrid_rag.stores.base import BaseVectorStoreManager
from hybrid_rag.stores.chroma_store import ChromaStoreManager
from hybrid_rag.stores.faiss_store import FaissStoreManager


def create_vector_store(
    backend: VectorStoreBackend | str,
    embeddings: Embeddings,
    settings: Settings,
    *,
    collection_name: str | None = None,
) -> BaseVectorStoreManager:
    backend_enum = VectorStoreBackend(backend) if isinstance(backend, str) else backend
    name = collection_name or settings.collection_name

    if backend_enum == VectorStoreBackend.FAISS:
        return FaissStoreManager(
            embeddings,
            index_dir=settings.faiss_index_dir,
            collection_name=name,
        )
    if backend_enum == VectorStoreBackend.CHROMA:
        return ChromaStoreManager(
            embeddings,
            persist_dir=settings.chroma_persist_dir,
            collection_name=name,
        )
    raise ConfigurationError(f"Unsupported vector store backend: {backend}")
