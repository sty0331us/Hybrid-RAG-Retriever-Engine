"""End-to-end RAG pipeline."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

from hybrid_rag.config.settings import (
    RetrieverStrategy,
    Settings,
    VectorStoreBackend,
    get_settings,
)
from hybrid_rag.core.exceptions import RAGPipelineError, StoreError
from hybrid_rag.core.logging import get_logger
from hybrid_rag.core.types import RAGAnswer, RetrievalResult
from hybrid_rag.ingestion import DocumentIngestor
from hybrid_rag.ingestion.loader import load_directory, load_file
from hybrid_rag.rag.llm import build_embeddings, build_llm
from hybrid_rag.rag.prompts import build_rag_prompt, format_context
from hybrid_rag.retrievers.factory import create_retriever
from hybrid_rag.retrievers.parent_document import ParentDocumentRetrieverStrategy
from hybrid_rag.stores.base import BaseVectorStoreManager
from hybrid_rag.stores.factory import create_vector_store

logger = get_logger(__name__)


class RAGEngine:
    """
    Production façade:
      ingest → index → retrieve (strategy) → generate → compare.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = build_llm(self.settings)
        self.embeddings = build_embeddings(self.settings)
        self.ingestor = DocumentIngestor(self.settings)
        self.prompt = build_rag_prompt()
        self._stores: dict[str, BaseVectorStoreManager] = {}

    def _store_key(self, backend: VectorStoreBackend | str, *, parent: bool = False) -> str:
        suffix = ":parent" if parent else ":flat"
        return f"{backend}{suffix}"

    def get_store(
        self,
        backend: VectorStoreBackend | str,
        *,
        parent: bool = False,
    ) -> BaseVectorStoreManager:
        key = self._store_key(backend, parent=parent)
        if key not in self._stores:
            collection = self.settings.collection_name
            if parent:
                collection = f"{collection}_parent"
            self._stores[key] = create_vector_store(
                backend,
                self.embeddings,
                self.settings,
                collection_name=collection,
            )
        return self._stores[key]

    def _load_raw_documents(self, source: Path) -> list[Document]:
        if source.is_dir():
            return load_directory(source)
        return load_file(source)

    def _try_load(self, store: BaseVectorStoreManager) -> bool:
        try:
            store.load()
            return True
        except StoreError:
            return False

    def ensure_store_loaded(
        self,
        backend: VectorStoreBackend | str,
        *,
        parent: bool = False,
    ) -> BaseVectorStoreManager:
        store = self.get_store(backend, parent=parent)
        try:
            _ = store.store
        except RuntimeError:
            store.load()
        return store

    def ingest_and_index(
        self,
        source: Path,
        *,
        backends: list[VectorStoreBackend] | None = None,
        rebuild: bool = True,
    ) -> dict[str, Any]:
        """Ingest documents and build flat + parent indexes for selected backends."""
        backends = backends or [VectorStoreBackend.FAISS, VectorStoreBackend.CHROMA]
        result = self.ingestor.ingest(source, for_parent_document=False)
        chunks, stats = result  # type: ignore[misc]
        raw_docs = self._load_raw_documents(source)

        payload: dict[str, Any] = {"ingest": stats.__dict__, "backends": {}}
        for backend in backends:
            flat_store = self.get_store(backend, parent=False)
            if rebuild:
                flat_store.delete_collection()
            if rebuild or not self._try_load(flat_store):
                flat_store.build(chunks)
            else:
                flat_store.add_documents(chunks)

            if not raw_docs:
                raise RAGPipelineError("No documents found to index")

            parent_store = self.get_store(backend, parent=True)
            if rebuild:
                parent_store.delete_collection()
                parent_store.create_empty()
            elif not self._try_load(parent_store):
                parent_store.create_empty()

            parent_strategy = ParentDocumentRetrieverStrategy(parent_store, self.settings)
            parent_strategy.add_documents(raw_docs)

            payload["backends"][str(backend)] = {
                "flat_chunks": len(chunks),
                "raw_documents": len(raw_docs),
            }
            logger.info("index_ready", backend=str(backend), flat_chunks=len(chunks))

        return payload

    def retrieve(
        self,
        query: str,
        *,
        strategy: RetrieverStrategy | str = RetrieverStrategy.VECTOR_STORE,
        backend: VectorStoreBackend | str | None = None,
    ) -> RetrievalResult:
        backend = backend or self.settings.default_vector_store
        strategy_enum = RetrieverStrategy(strategy) if isinstance(strategy, str) else strategy
        use_parent = strategy_enum == RetrieverStrategy.PARENT_DOCUMENT
        store = self.ensure_store_loaded(backend, parent=use_parent)
        retriever = create_retriever(strategy_enum, store, self.settings, llm=self.llm)
        result = retriever.retrieve(query)
        logger.info(
            "retrieve_done",
            strategy=result.strategy,
            backend=result.vector_store,
            chunks=len(result.chunks),
            latency_ms=round(result.latency_ms, 2),
        )
        return result

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _generate(self, question: str, context: str) -> str:
        messages = self.prompt.format_messages(context=context, question=question)
        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            return "".join(str(part) for part in content)
        return str(content)

    def ask(
        self,
        query: str,
        *,
        strategy: RetrieverStrategy | str = RetrieverStrategy.VECTOR_STORE,
        backend: VectorStoreBackend | str | None = None,
    ) -> RAGAnswer:
        backend = backend or self.settings.default_vector_store
        try:
            retrieval = self.retrieve(query, strategy=strategy, backend=backend)
            context = format_context(retrieval.chunks)
            gen_started = time.perf_counter()
            answer = self._generate(query, context)
            gen_ms = (time.perf_counter() - gen_started) * 1000
            return RAGAnswer(
                query=query,
                answer=answer,
                strategy=retrieval.strategy,
                vector_store=retrieval.vector_store,
                sources=retrieval.chunks,
                retrieval_latency_ms=retrieval.latency_ms,
                generation_latency_ms=gen_ms,
                total_latency_ms=retrieval.latency_ms + gen_ms,
                model=self.settings.llm_model,
            )
        except Exception as exc:  # noqa: BLE001
            raise RAGPipelineError(f"RAG ask failed: {exc}", details={"query": query}) from exc

    def compare_strategies(
        self,
        query: str,
        *,
        strategies: list[RetrieverStrategy] | None = None,
        backend: VectorStoreBackend | str | None = None,
        generate_answers: bool = True,
    ) -> list[RAGAnswer | RetrievalResult]:
        strategies = strategies or list(RetrieverStrategy)
        backend = backend or self.settings.default_vector_store
        outputs: list[RAGAnswer | RetrievalResult] = []
        for strategy in strategies:
            try:
                if generate_answers:
                    outputs.append(self.ask(query, strategy=strategy, backend=backend))
                else:
                    outputs.append(self.retrieve(query, strategy=strategy, backend=backend))
            except Exception as exc:  # noqa: BLE001
                logger.exception("strategy_failed", strategy=str(strategy), error=str(exc))
                outputs.append(
                    RetrievalResult(
                        query=query,
                        strategy=str(strategy),
                        vector_store=str(backend),
                        chunks=[],
                        latency_ms=0.0,
                        extras={"error": str(exc)},
                    )
                )
        return outputs


@lru_cache(maxsize=1)
def get_engine() -> RAGEngine:
    return RAGEngine()
