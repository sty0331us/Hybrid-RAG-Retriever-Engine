"""Parent document retriever: retrieve small chunks, return larger parent context."""

from __future__ import annotations

from pathlib import Path

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_rag.config.settings import Settings
from hybrid_rag.core.logging import get_logger
from hybrid_rag.retrievers.base import BaseStrategyRetriever
from hybrid_rag.stores.base import BaseVectorStoreManager

logger = get_logger(__name__)


class ParentDocumentRetrieverStrategy(BaseStrategyRetriever):
    """
    Indexes small child chunks for precise matching, returns parent docs for generation.

    Best for: long documents where context around a match matters.
    Trade-off: requires a byte/doc store for parents; indexing is more involved.
    """

    strategy_name = "parent_document"

    def __init__(
        self,
        store: BaseVectorStoreManager,
        settings: Settings,
        *,
        top_k: int = 4,
        docstore_path: Path | None = None,
    ) -> None:
        super().__init__(store.backend_name, top_k=top_k)
        self.store = store
        self.settings = settings
        self.docstore_path = docstore_path or (settings.docstore_dir / store.backend_name)
        self.docstore_path.mkdir(parents=True, exist_ok=True)
        self._byte_store = LocalFileStore(str(self.docstore_path))
        self._docstore = create_kv_docstore(self._byte_store)

    def build(self) -> BaseRetriever:
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.child_chunk_size,
            chunk_overlap=max(20, self.settings.chunk_overlap // 2),
        )
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.parent_chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self._retriever = ParentDocumentRetriever(
            vectorstore=self.store.store,
            docstore=self._docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": self.top_k},
        )
        logger.info(
            "parent_document_retriever_ready",
            top_k=self.top_k,
            docstore=str(self.docstore_path),
        )
        return self._retriever

    def add_documents(self, documents) -> None:  # type: ignore[no-untyped-def]
        """Index parents into docstore and children into the vector store."""
        retriever = self.retriever
        assert isinstance(retriever, ParentDocumentRetriever)
        retriever.add_documents(documents, ids=None)
        self.store.persist()
        logger.info("parent_documents_indexed", count=len(documents))
