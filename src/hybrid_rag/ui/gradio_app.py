"""Gradio UI — production operator console for strategy comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from hybrid_rag.config.settings import RetrieverStrategy, Settings, VectorStoreBackend, get_settings
from hybrid_rag.core.logging import get_logger
from hybrid_rag.core.types import RAGAnswer, RetrievalResult
from hybrid_rag.evaluation.benchmark import RetrieverBenchmark
from hybrid_rag.rag.pipeline import RAGEngine

logger = get_logger(__name__)

STRATEGY_CHOICES = [s.value for s in RetrieverStrategy]
BACKEND_CHOICES = [b.value for b in VectorStoreBackend]


def _format_sources(answer: RAGAnswer | RetrievalResult) -> str:
    chunks = answer.sources if isinstance(answer, RAGAnswer) else answer.chunks
    if not chunks:
        extras = getattr(answer, "extras", {}) or {}
        if extras.get("error"):
            return f"Error: {extras['error']}"
        return "_No chunks retrieved._"
    blocks: list[str] = []
    for c in chunks:
        title = c.metadata.get("title") or c.metadata.get("source") or "unknown"
        cat = c.metadata.get("category", "general")
        score = f"{c.score:.4f}" if c.score is not None else "n/a"
        preview = c.content[:500] + ("…" if len(c.content) > 500 else "")
        blocks.append(
            f"**#{c.rank}** · `{cat}` · {title} · score={score}\n\n{preview}"
        )
    return "\n\n---\n\n".join(blocks)


def _benchmark_table(rows: list[dict[str, Any]]) -> list[list[Any]]:
    table = []
    for r in rows:
        table.append(
            [
                r.get("strategy"),
                r.get("vector_store"),
                r.get("latency_ms"),
                r.get("num_chunks"),
                r.get("avg_score"),
                r.get("unique_sources"),
                (r.get("answer_preview") or "")[:120],
                r.get("notes") or "",
            ]
        )
    return table


def build_ui(engine: RAGEngine | None = None, settings: Settings | None = None) -> gr.Blocks:
    settings = settings or get_settings()
    engine = engine or RAGEngine(settings)
    bench = RetrieverBenchmark(engine)

    with gr.Blocks(
        title="Hybrid RAG Retriever Engine",
        theme=gr.themes.Soft(primary_hue="slate", secondary_hue="zinc"),
        css="""
        .title { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }
        .subtitle { color: #64748b; margin-bottom: 1rem; }
        """,
    ) as demo:
        gr.Markdown(
            """
<div class="title">Hybrid RAG Retriever Engine</div>
<div class="subtitle">
Compare vector-store, multi-query, self-query, parent-document, and ensemble
(hybrid dense+lexical RRF) retrievers across FAISS and Chroma.
</div>
"""
        )

        with gr.Tab("Ask"):
            with gr.Row():
                strategy = gr.Dropdown(
                    STRATEGY_CHOICES,
                    value=settings.default_retriever.value,
                    label="Retriever strategy",
                )
                backend = gr.Dropdown(
                    BACKEND_CHOICES,
                    value=settings.default_vector_store.value,
                    label="Vector store",
                )
            query = gr.Textbox(
                label="Question",
                lines=3,
                placeholder="e.g. Compare FAISS and Chroma for production RAG workloads",
            )
            ask_btn = gr.Button("Run RAG", variant="primary")
            answer_out = gr.Markdown(label="Answer")
            meta_out = gr.JSON(label="Latency / metadata")
            sources_out = gr.Markdown(label="Retrieved sources")

            def _ask(q: str, strat: str, store: str):
                if not q.strip():
                    return "Enter a question.", {}, ""
                result = engine.ask(q, strategy=strat, backend=store)
                meta = {
                    "strategy": result.strategy,
                    "vector_store": result.vector_store,
                    "model": result.model,
                    "retrieval_latency_ms": round(result.retrieval_latency_ms, 2),
                    "generation_latency_ms": round(result.generation_latency_ms, 2),
                    "total_latency_ms": round(result.total_latency_ms, 2),
                    "num_sources": len(result.sources),
                }
                return result.answer, meta, _format_sources(result)

            ask_btn.click(_ask, [query, strategy, backend], [answer_out, meta_out, sources_out])

        with gr.Tab("Compare Retrievers"):
            gr.Markdown(
                "Runs **all** retrieval strategies against the same query and vector backend, "
                "then ranks by latency and source diversity."
            )
            cmp_query = gr.Textbox(label="Question", lines=2)
            cmp_backend = gr.Dropdown(BACKEND_CHOICES, value="faiss", label="Vector store")
            gen_answers = gr.Checkbox(value=True, label="Generate LLM answers (slower, richer)")
            cmp_btn = gr.Button("Compare strategies", variant="primary")
            cmp_summary = gr.JSON(label="Summary")
            cmp_table = gr.Dataframe(
                headers=[
                    "strategy",
                    "vector_store",
                    "latency_ms",
                    "num_chunks",
                    "avg_score",
                    "unique_sources",
                    "answer_preview",
                    "notes",
                ],
                label="Benchmark results",
                wrap=True,
            )
            cmp_detail = gr.Markdown()

            def _compare(q: str, store: str, with_answers: bool):
                if not q.strip():
                    return {}, [], "Enter a question."
                benchmarks, summary = bench.compare_retrievers(
                    q, backend=store, generate_answers=with_answers
                )
                detail_parts = []
                for b in benchmarks:
                    detail_parts.append(
                        f"### `{b.strategy}` ({b.latency_ms} ms)\n"
                        f"chunks={b.num_chunks} · unique_sources={b.unique_sources}\n\n"
                        f"{b.answer_preview or '_retrieval only_'}"
                    )
                return summary, _benchmark_table([b.model_dump() for b in benchmarks]), "\n\n".join(
                    detail_parts
                )

            cmp_btn.click(
                _compare,
                [cmp_query, cmp_backend, gen_answers],
                [cmp_summary, cmp_table, cmp_detail],
            )

        with gr.Tab("Compare Vector Stores"):
            gr.Markdown("Head-to-head **FAISS vs Chroma** on the same strategy and query.")
            vs_query = gr.Textbox(label="Question", lines=2)
            vs_strategy = gr.Dropdown(
                STRATEGY_CHOICES, value="vector_store", label="Retriever strategy"
            )
            vs_btn = gr.Button("Compare FAISS vs Chroma", variant="primary")
            vs_summary = gr.JSON(label="Summary")
            vs_table = gr.Dataframe(
                headers=[
                    "strategy",
                    "vector_store",
                    "latency_ms",
                    "num_chunks",
                    "avg_score",
                    "unique_sources",
                    "answer_preview",
                    "notes",
                ],
                label="Store comparison",
                wrap=True,
            )

            def _compare_stores(q: str, strat: str):
                if not q.strip():
                    return {}, []
                benchmarks, summary = bench.compare_vector_stores(
                    q, strategy=strat, generate_answers=False
                )
                return summary, _benchmark_table([b.model_dump() for b in benchmarks])

            vs_btn.click(_compare_stores, [vs_query, vs_strategy], [vs_summary, vs_table])

        with gr.Tab("Ingest"):
            gr.Markdown(
                f"Index documents from a directory (default: `{settings.data_dir / 'sample'}`). "
                "Builds both flat and parent-document indexes for FAISS and Chroma."
            )
            path_in = gr.Textbox(
                value=str(settings.data_dir / "sample"),
                label="Source path",
            )
            rebuild = gr.Checkbox(value=True, label="Rebuild indexes from scratch")
            ingest_btn = gr.Button("Ingest & index", variant="primary")
            ingest_out = gr.JSON(label="Ingest report")

            def _ingest(path: str, do_rebuild: bool):
                report = engine.ingest_and_index(Path(path), rebuild=do_rebuild)
                return report

            ingest_btn.click(_ingest, [path_in, rebuild], [ingest_out])

        with gr.Tab("Architecture"):
            gr.Markdown(
                """
## Retriever strategies

| Strategy | How it works | When to use |
|---|---|---|
| **vector_store** | Dense similarity over chunks | Low-latency baseline |
| **multi_query** | LLM expands query; unions hits | Ambiguous questions |
| **self_query** | Semantic query + metadata filters | Constrained queries |
| **parent_document** | Match children, return parents | Long-document context |
| **ensemble** | Dense + lexical RRF fusion | Keyword + semantic mix |

## Vector stores

| Store | Strengths | Trade-offs |
|---|---|---|
| **FAISS** | Fast in-process similarity search | You own persistence/filtering |
| **Chroma** | Persistence + metadata filters | Heavier dependency profile |

## Production notes

- Config via pydantic-settings / `.env`
- Structured logging (structlog)
- Typed domain models for answers & benchmarks
- Retry-wrapped LLM calls
- Isolated flat vs parent indexes per backend
"""
            )

    return demo


def launch(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    demo = build_ui(settings=settings)
    logger.info(
        "gradio_launch",
        host=settings.gradio_server_name,
        port=settings.gradio_server_port,
    )
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        share=settings.gradio_share,
    )
