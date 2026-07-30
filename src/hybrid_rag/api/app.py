"""FastAPI application exposing RAG operations over HTTP."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from hybrid_rag.api.schemas import (
    AskRequest,
    AskResponse,
    ChunkResponse,
    CompareRequest,
    CompareResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from hybrid_rag.config.settings import RetrieverStrategy, Settings, VectorStoreBackend, get_settings
from hybrid_rag.core.exceptions import HybridRAGError
from hybrid_rag.core.logging import configure_logging, get_logger
from hybrid_rag.evaluation.benchmark import RetrieverBenchmark
from hybrid_rag.rag.pipeline import RAGEngine

logger = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    engine: RAGEngine | None = None,
) -> FastAPI:
    """Build a FastAPI app. Inject engine in tests to avoid real LLM/store I/O."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    app = FastAPI(
        title="Hybrid RAG Retriever Engine",
        description="REST API for multi-strategy RAG retrieval, generation, and benchmarking.",
        version="1.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_engine() -> RAGEngine:
        return engine or RAGEngine(settings)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_env=settings.app_env.value,
            default_retriever=settings.default_retriever.value,
            default_vector_store=settings.default_vector_store.value,
            strategies=[s.value for s in RetrieverStrategy],
            vector_stores=[b.value for b in VectorStoreBackend],
        )

    @app.post("/ask", response_model=AskResponse, tags=["rag"])
    def ask(body: AskRequest, rag: RAGEngine = Depends(get_engine)) -> AskResponse:
        try:
            answer = rag.ask(body.query, strategy=body.strategy, backend=body.backend)
        except HybridRAGError as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("api_ask_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return AskResponse(
            query=answer.query,
            answer=answer.answer,
            strategy=answer.strategy,
            vector_store=answer.vector_store,
            sources=[ChunkResponse(**c.model_dump()) for c in answer.sources],
            retrieval_latency_ms=answer.retrieval_latency_ms,
            generation_latency_ms=answer.generation_latency_ms,
            total_latency_ms=answer.total_latency_ms,
            model=answer.model,
        )

    @app.post("/retrieve", response_model=RetrieveResponse, tags=["rag"])
    def retrieve(body: RetrieveRequest, rag: RAGEngine = Depends(get_engine)) -> RetrieveResponse:
        try:
            result = rag.retrieve(body.query, strategy=body.strategy, backend=body.backend)
        except HybridRAGError as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("api_retrieve_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return RetrieveResponse(
            query=result.query,
            strategy=result.strategy,
            vector_store=result.vector_store,
            chunks=[ChunkResponse(**c.model_dump()) for c in result.chunks],
            latency_ms=result.latency_ms,
            extras=result.extras,
        )

    @app.post("/compare/retrievers", response_model=CompareResponse, tags=["benchmark"])
    def compare_retrievers(
        body: CompareRequest,
        rag: RAGEngine = Depends(get_engine),
    ) -> CompareResponse:
        try:
            benchmarks, summary = RetrieverBenchmark(rag).compare_retrievers(
                body.query,
                backend=body.backend,
                generate_answers=body.generate_answers,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("api_compare_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return CompareResponse(
            summary=summary,
            benchmarks=[b.model_dump() for b in benchmarks],
        )

    @app.post("/ingest", response_model=IngestResponse, tags=["ingest"])
    def ingest(body: IngestRequest, rag: RAGEngine = Depends(get_engine)) -> IngestResponse:
        source = Path(body.source)
        if not source.exists():
            raise HTTPException(status_code=404, detail=f"Source not found: {body.source}")
        try:
            report = rag.ingest_and_index(source, backends=body.backends, rebuild=body.rebuild)
        except HybridRAGError as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("api_ingest_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return IngestResponse(report=report)

    return app
