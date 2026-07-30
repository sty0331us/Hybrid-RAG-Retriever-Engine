"""API unit tests with an injected fake engine (no network / OpenAI)."""

from __future__ import annotations

from hybrid_rag.api.app import create_app
from hybrid_rag.config.settings import Settings
from hybrid_rag.core.types import RAGAnswer, RetrievalResult, RetrievedChunk


class _FakeEngine:
    def ask(self, query: str, *, strategy=None, backend=None) -> RAGAnswer:
        return RAGAnswer(
            query=query,
            answer="Grounded answer.",
            strategy=str(strategy),
            vector_store=str(backend),
            sources=[
                RetrievedChunk(content="chunk", metadata={"source": "a.md"}, score=0.9, rank=1)
            ],
            retrieval_latency_ms=1.0,
            generation_latency_ms=2.0,
            total_latency_ms=3.0,
            model="test-model",
        )

    def retrieve(self, query: str, *, strategy=None, backend=None) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            strategy=str(strategy),
            vector_store=str(backend),
            chunks=[
                RetrievedChunk(content="chunk", metadata={"source": "a.md"}, score=0.9, rank=1)
            ],
            latency_ms=1.5,
            extras={"fake": True},
        )

    def ingest_and_index(self, source, *, backends=None, rebuild=True):
        return {"ok": True, "source": str(source), "rebuild": rebuild}


def _client():
    from fastapi.testclient import TestClient

    settings = Settings(openai_api_key="sk-test")  # type: ignore[arg-type]
    app = create_app(settings=settings, engine=_FakeEngine())  # type: ignore[arg-type]
    return TestClient(app)


def test_health_endpoint():
    client = _client()
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "ensemble" in body["strategies"]
    assert "faiss" in body["vector_stores"]


def test_ask_endpoint():
    client = _client()
    resp = client.post(
        "/ask",
        json={"query": "What is FAISS?", "strategy": "vector_store", "backend": "faiss"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Grounded answer."
    assert body["total_latency_ms"] == 3.0
    assert len(body["sources"]) == 1


def test_retrieve_endpoint():
    client = _client()
    resp = client.post(
        "/retrieve",
        json={"query": "hybrid retrieval", "strategy": "ensemble", "backend": "chroma"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "ensemble"
    assert body["extras"]["fake"] is True


def test_ingest_missing_source():
    client = _client()
    resp = client.post("/ingest", json={"source": "/tmp/does-not-exist-hybrid-rag"})
    assert resp.status_code == 404
