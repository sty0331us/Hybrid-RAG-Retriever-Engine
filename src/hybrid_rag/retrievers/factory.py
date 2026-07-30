"""Retriever strategy factory."""

from __future__ import annotations

from langchain_core.language_models import BaseLanguageModel

from hybrid_rag.config.settings import RetrieverStrategy, Settings
from hybrid_rag.core.exceptions import ConfigurationError
from hybrid_rag.retrievers.base import BaseStrategyRetriever
from hybrid_rag.retrievers.ensemble import EnsembleHybridRetrieverStrategy
from hybrid_rag.retrievers.multi_query import MultiQueryRetrieverStrategy
from hybrid_rag.retrievers.parent_document import ParentDocumentRetrieverStrategy
from hybrid_rag.retrievers.self_query import SelfQueryRetrieverStrategy
from hybrid_rag.retrievers.vector_store import VectorStoreRetrieverStrategy
from hybrid_rag.stores.base import BaseVectorStoreManager


def create_retriever(
    strategy: RetrieverStrategy | str,
    store: BaseVectorStoreManager,
    settings: Settings,
    *,
    llm: BaseLanguageModel | None = None,
) -> BaseStrategyRetriever:
    strategy_enum = RetrieverStrategy(strategy) if isinstance(strategy, str) else strategy
    top_k = settings.top_k

    if strategy_enum == RetrieverStrategy.VECTOR_STORE:
        return VectorStoreRetrieverStrategy(store, top_k=top_k)

    if strategy_enum == RetrieverStrategy.MULTI_QUERY:
        if llm is None:
            raise ConfigurationError("multi_query strategy requires an LLM")
        return MultiQueryRetrieverStrategy(store, llm, top_k=top_k)

    if strategy_enum == RetrieverStrategy.SELF_QUERY:
        if llm is None:
            raise ConfigurationError("self_query strategy requires an LLM")
        return SelfQueryRetrieverStrategy(store, llm, top_k=top_k)

    if strategy_enum == RetrieverStrategy.PARENT_DOCUMENT:
        return ParentDocumentRetrieverStrategy(store, settings, top_k=top_k)

    if strategy_enum == RetrieverStrategy.ENSEMBLE:
        return EnsembleHybridRetrieverStrategy(store, top_k=top_k)

    raise ConfigurationError(f"Unsupported retriever strategy: {strategy}")
