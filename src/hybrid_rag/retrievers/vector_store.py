"""Classic vector-store-backed retriever."""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever

from hybrid_rag.retrievers.base import BaseStrategyRetriever
from hybrid_rag.stores.base import BaseVectorStoreManager


class VectorStoreRetrieverStrategy(BaseStrategyRetriever):
    """
    Baseline semantic similarity retriever backed by FAISS or Chroma.

    Best for: straightforward semantic search with low latency.
    Trade-off: sensitive to query phrasing; no query expansion or metadata filters.
    """

    strategy_name = "vector_store"

    def __init__(
        self,
        store: BaseVectorStoreManager,
        *,
        top_k: int = 4,
        search_type: str = "similarity",
        score_threshold: float | None = None,
    ) -> None:
        super().__init__(store.backend_name, top_k=top_k)
        self.store = store
        self.search_type = search_type
        self.score_threshold = score_threshold

    def build(self) -> BaseRetriever:
        search_kwargs: dict = {"k": self.top_k}
        if self.search_type == "similarity_score_threshold" and self.score_threshold is not None:
            search_kwargs["score_threshold"] = self.score_threshold
        self._retriever = self.store.as_retriever(
            search_type=self.search_type,
            search_kwargs=search_kwargs,
        )
        return self._retriever
