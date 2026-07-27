"""Side-by-side retriever strategy + vector-store benchmarking."""

from __future__ import annotations

from hybrid_rag.config.settings import RetrieverStrategy, VectorStoreBackend
from hybrid_rag.core.logging import get_logger
from hybrid_rag.core.types import StrategyBenchmark
from hybrid_rag.evaluation.metrics import summarize_comparison, to_benchmark
from hybrid_rag.rag.pipeline import RAGEngine

logger = get_logger(__name__)


class RetrieverBenchmark:
    """Runs controlled comparisons suitable for hiring-portfolio demos & CI smoke tests."""

    def __init__(self, engine: RAGEngine) -> None:
        self.engine = engine

    def compare_retrievers(
        self,
        query: str,
        *,
        backend: VectorStoreBackend | str = VectorStoreBackend.FAISS,
        strategies: list[RetrieverStrategy] | None = None,
        generate_answers: bool = True,
    ) -> tuple[list[StrategyBenchmark], dict]:
        results = self.engine.compare_strategies(
            query,
            strategies=strategies,
            backend=backend,
            generate_answers=generate_answers,
        )
        benchmarks = [to_benchmark(r) for r in results]
        summary = summarize_comparison(benchmarks)
        logger.info(
            "benchmark_retrievers_done",
            query=query,
            backend=str(backend),
            fastest=summary.get("fastest_strategy"),
        )
        return benchmarks, summary

    def compare_vector_stores(
        self,
        query: str,
        *,
        strategy: RetrieverStrategy | str = RetrieverStrategy.VECTOR_STORE,
        backends: list[VectorStoreBackend] | None = None,
        generate_answers: bool = False,
    ) -> tuple[list[StrategyBenchmark], dict]:
        backends = backends or [VectorStoreBackend.FAISS, VectorStoreBackend.CHROMA]
        benchmarks: list[StrategyBenchmark] = []
        for backend in backends:
            try:
                if generate_answers:
                    result = self.engine.ask(query, strategy=strategy, backend=backend)
                else:
                    result = self.engine.retrieve(query, strategy=strategy, backend=backend)
                benchmarks.append(to_benchmark(result))
            except Exception as exc:  # noqa: BLE001
                logger.exception("vector_store_bench_failed", backend=str(backend))
                benchmarks.append(
                    StrategyBenchmark(
                        strategy=str(strategy),
                        vector_store=str(backend),
                        query=query,
                        latency_ms=0.0,
                        num_chunks=0,
                        notes=str(exc),
                    )
                )
        return benchmarks, summarize_comparison(benchmarks)
