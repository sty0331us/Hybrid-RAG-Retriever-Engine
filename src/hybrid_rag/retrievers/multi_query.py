"""Multi-query retriever: LLM expands one query into several perspectives."""

from __future__ import annotations

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever

from hybrid_rag.core.logging import get_logger
from hybrid_rag.retrievers.base import BaseStrategyRetriever
from hybrid_rag.stores.base import BaseVectorStoreManager

logger = get_logger(__name__)


class MultiQueryRetrieverStrategy(BaseStrategyRetriever):
    """
    Generates multiple query variants via LLM, retrieves for each, then unions results.

    Best for: ambiguous / underspecified user questions.
    Trade-off: higher latency and LLM cost vs plain vector retrieval.
    """

    strategy_name = "multi_query"

    def __init__(
        self,
        store: BaseVectorStoreManager,
        llm: BaseLanguageModel,
        *,
        top_k: int = 4,
        include_original: bool = True,
    ) -> None:
        super().__init__(store.backend_name, top_k=top_k)
        self.store = store
        self.llm = llm
        self.include_original = include_original

    def build(self) -> BaseRetriever:
        base = self.store.as_retriever(search_kwargs={"k": self.top_k})
        self._retriever = MultiQueryRetriever.from_llm(
            retriever=base,
            llm=self.llm,
            include_original=self.include_original,
        )
        logger.info("multi_query_retriever_ready", top_k=self.top_k)
        return self._retriever
