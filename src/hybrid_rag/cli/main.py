"""Typer CLI for ingest, ask, compare, and UI launch."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from hybrid_rag.config.settings import RetrieverStrategy, VectorStoreBackend, get_settings
from hybrid_rag.core.logging import configure_logging, get_logger

app = typer.Typer(
    name="hybrid-rag",
    help="Production Hybrid RAG Retriever Engine",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
logger = get_logger(__name__)


def _boot() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)


@app.command("ingest")
def ingest_cmd(
    source: Path = typer.Argument(..., exists=True, help="File or directory to ingest"),
    backends: str = typer.Option("faiss,chroma", help="Comma-separated: faiss,chroma"),
    rebuild: bool = typer.Option(True, help="Rebuild indexes from scratch"),
) -> None:
    """Ingest documents and build vector indexes."""
    _boot()
    from hybrid_rag.rag.pipeline import RAGEngine

    engine = RAGEngine()
    backend_list = [VectorStoreBackend(b.strip()) for b in backends.split(",") if b.strip()]
    report = engine.ingest_and_index(source, backends=backend_list, rebuild=rebuild)
    console.print_json(json.dumps(report, default=str))


@app.command("ask")
def ask_cmd(
    query: str = typer.Argument(..., help="Natural language question"),
    strategy: RetrieverStrategy = typer.Option(RetrieverStrategy.VECTOR_STORE),
    backend: VectorStoreBackend = typer.Option(VectorStoreBackend.FAISS),
) -> None:
    """Run a single RAG query."""
    _boot()
    from hybrid_rag.rag.pipeline import RAGEngine

    answer = RAGEngine().ask(query, strategy=strategy, backend=backend)
    console.print(f"\n[bold]Answer[/bold] ({answer.strategy} / {answer.vector_store})")
    console.print(answer.answer)
    console.print(
        f"\n[dim]retrieval={answer.retrieval_latency_ms:.1f}ms  "
        f"generation={answer.generation_latency_ms:.1f}ms  "
        f"total={answer.total_latency_ms:.1f}ms[/dim]"
    )


@app.command("compare-retrievers")
def compare_retrievers_cmd(
    query: str = typer.Argument(...),
    backend: VectorStoreBackend = typer.Option(VectorStoreBackend.FAISS),
    answers: bool = typer.Option(True, help="Also generate LLM answers"),
) -> None:
    """Benchmark all retriever strategies on one query."""
    _boot()
    from hybrid_rag.evaluation.benchmark import RetrieverBenchmark
    from hybrid_rag.rag.pipeline import RAGEngine

    benchmarks, summary = RetrieverBenchmark(RAGEngine()).compare_retrievers(
        query, backend=backend, generate_answers=answers
    )
    table = Table(title="Retriever strategy comparison")
    table.add_column("Strategy")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Unique sources", justify="right")
    table.add_column("Notes")
    for b in benchmarks:
        table.add_row(
            b.strategy,
            f"{b.latency_ms:.1f}",
            str(b.num_chunks),
            str(b.unique_sources),
            b.notes[:60],
        )
    console.print(table)
    console.print_json(json.dumps(summary, default=str))


@app.command("compare-stores")
def compare_stores_cmd(
    query: str = typer.Argument(...),
    strategy: RetrieverStrategy = typer.Option(RetrieverStrategy.VECTOR_STORE),
) -> None:
    """Benchmark FAISS vs Chroma on one strategy."""
    _boot()
    from hybrid_rag.evaluation.benchmark import RetrieverBenchmark
    from hybrid_rag.rag.pipeline import RAGEngine

    benchmarks, summary = RetrieverBenchmark(RAGEngine()).compare_vector_stores(
        query, strategy=strategy, generate_answers=False
    )
    table = Table(title="Vector store comparison")
    table.add_column("Store")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Notes")
    for b in benchmarks:
        table.add_row(b.vector_store, f"{b.latency_ms:.1f}", str(b.num_chunks), b.notes[:60])
    console.print(table)
    console.print_json(json.dumps(summary, default=str))


@app.command("ui")
def ui_cmd(
    host: str | None = typer.Option(None, help="Bind address"),
    port: int | None = typer.Option(None, help="Bind port"),
) -> None:
    """Launch the Gradio operator console."""
    _boot()
    settings = get_settings()
    if host:
        settings.gradio_server_name = host
    if port:
        settings.gradio_server_port = port
    from hybrid_rag.ui.gradio_app import launch

    launch(settings)


@app.command("serve")
def serve_cmd(
    host: str | None = typer.Option(None, help="API bind address"),
    port: int | None = typer.Option(None, help="API bind port"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes"),
) -> None:
    """Launch the FastAPI REST service."""
    _boot()
    settings = get_settings()
    bind_host = host or settings.api_host
    bind_port = port or settings.api_port
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        console.print("[red]Missing API dependencies.[/red] Install with: pip install -e '.[api]'")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]Serving API[/bold] at http://{bind_host}:{bind_port}")
    console.print("Docs: /docs  ·  Health: /health")
    uvicorn.run(
        "hybrid_rag.api.app:create_app",
        host=bind_host,
        port=bind_port,
        reload=reload,
        factory=True,
    )


@app.callback()
def main_callback() -> None:
    """Hybrid RAG Retriever Engine CLI."""


# Typer object exported as console script entry `hybrid-rag`
# Also allow: python -m hybrid_rag
def main() -> None:
    app()


if __name__ == "__main__":
    main()
