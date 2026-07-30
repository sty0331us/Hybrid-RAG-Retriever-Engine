"""Ensemble hybrid retriever: dense + lexical fusion via Reciprocal Rank Fusion."""

from __future__ import annotations

import re
import time
from collections import Counter

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from hybrid_rag.core.types import RetrievalResult, RetrievedChunk
from hybrid_rag.retrievers.base import BaseStrategyRetriever
from hybrid_rag.stores.base import BaseVectorStoreManager

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def lexical_score(query: str, content: str) -> float:
    """Lightweight BM25-ish score without an external dependency."""
    q_tokens = tokenize(query)
    d_tokens = tokenize(content)
    if not q_tokens or not d_tokens:
        return 0.0

    tf = Counter(d_tokens)
    dl = len(d_tokens)
    avgdl = max(dl, 1)
    k1, b = 1.5, 0.75
    score = 0.0
    for term in set(q_tokens):
        freq = tf.get(term, 0)
        if freq == 0:
            continue
        # Uniform IDF for candidate-pool scoring; relative ordering still works.
        idf = 1.5
        denom = freq + k1 * (1 - b + b * dl / avgdl)
        score += idf * (freq * (k1 + 1)) / denom
    return score


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked id lists with Reciprocal Rank Fusion.

    Returns (doc_id, rrf_score) sorted descending by score.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _doc_id(doc: Document, fallback: int) -> str:
    meta = doc.metadata or {}
    for key in ("doc_id", "id", "source", "title"):
        value = meta.get(key)
        if value:
            return f"{value}:{meta.get('chunk_index', fallback)}"
    return f"anon:{fallback}:{hash(doc.page_content) & 0xFFFFFFFF:x}"


class EnsembleHybridRetrieverStrategy(BaseStrategyRetriever):
    """
    Hybrid dense + lexical retrieval fused with Reciprocal Rank Fusion (RRF).

    Best for: queries that mix semantic intent with exact keywords / jargon.
    Trade-off: slightly higher latency than pure vector search; no extra LLM cost.
    """

    strategy_name = "ensemble"

    def __init__(
        self,
        store: BaseVectorStoreManager,
        *,
        top_k: int = 4,
        fetch_k: int | None = None,
        rrf_k: int = 60,
    ) -> None:
        super().__init__(store.backend_name, top_k=top_k)
        self.store = store
        self.fetch_k = fetch_k or max(top_k * 4, 12)
        self.rrf_k = rrf_k

    def build(self) -> BaseRetriever:
        self._retriever = self.store.as_retriever(search_kwargs={"k": self.fetch_k})
        return self._retriever

    def retrieve(self, query: str, **kwargs: object) -> RetrievalResult:
        started = time.perf_counter()
        dense_docs = self.store.similarity_search(query, k=self.fetch_k)
        if not dense_docs:
            return RetrievalResult(
                query=query,
                strategy=self.strategy_name,
                vector_store=self.vector_store_name,
                chunks=[],
                latency_ms=(time.perf_counter() - started) * 1000,
                extras={"fusion": "rrf", "dense_hits": 0, "lexical_hits": 0},
            )

        id_to_doc: dict[str, Document] = {}
        dense_ranking: list[str] = []
        for i, doc in enumerate(dense_docs):
            doc_id = _doc_id(doc, i)
            id_to_doc[doc_id] = doc
            dense_ranking.append(doc_id)

        lexical_ranked = sorted(
            id_to_doc.items(),
            key=lambda item: lexical_score(query, item[1].page_content),
            reverse=True,
        )
        lexical_ranking = [doc_id for doc_id, _ in lexical_ranked]

        fused = reciprocal_rank_fusion(
            [dense_ranking, lexical_ranking],
            k=self.rrf_k,
        )[: self.top_k]

        chunks = [
            RetrievedChunk(
                content=id_to_doc[doc_id].page_content,
                metadata={
                    **dict(id_to_doc[doc_id].metadata),
                    "rrf_score": round(score, 6),
                    "fusion": "dense+lexical_rrf",
                },
                score=round(score, 6),
                rank=i + 1,
            )
            for i, (doc_id, score) in enumerate(fused)
        ]

        latency_ms = (time.perf_counter() - started) * 1000
        return RetrievalResult(
            query=query,
            strategy=self.strategy_name,
            vector_store=self.vector_store_name,
            chunks=chunks,
            latency_ms=latency_ms,
            extras={
                "fusion": "rrf",
                "dense_hits": len(dense_ranking),
                "lexical_hits": len(lexical_ranking),
                "fetch_k": self.fetch_k,
            },
        )
