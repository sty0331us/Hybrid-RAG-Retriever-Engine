"""Ingestion orchestration."""

from __future__ import annotations

import time
from pathlib import Path

from langchain_core.documents import Document

from hybrid_rag.config.settings import Settings
from hybrid_rag.core.logging import get_logger
from hybrid_rag.core.types import IngestStats
from hybrid_rag.ingestion.chunker import chunk_documents, chunk_parent_child
from hybrid_rag.ingestion.loader import load_directory, load_paths

logger = get_logger(__name__)


class DocumentIngestor:
    """Load → normalize → chunk documents for indexing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def ingest(
        self,
        source: Path | list[Path],
        *,
        for_parent_document: bool = False,
    ) -> tuple[list[Document], IngestStats] | tuple[list[Document], list[Document], IngestStats]:
        started = time.perf_counter()
        stats = IngestStats()
        paths = [source] if isinstance(source, Path) else list(source)

        raw_docs: list[Document] = []
        if len(paths) == 1 and paths[0].is_dir():
            raw_docs = load_directory(paths[0])
        else:
            raw_docs = load_paths(paths)

        stats.documents_loaded = len(raw_docs)
        stats.files_processed = len({d.metadata.get("source") for d in raw_docs})

        if for_parent_document:
            parents, children = chunk_parent_child(raw_docs, self.settings)
            stats.chunks_created = len(children)
            stats.duration_ms = (time.perf_counter() - started) * 1000
            logger.info("ingest_complete", mode="parent_child", **stats.__dict__)
            return parents, children, stats

        chunks = chunk_documents(raw_docs, self.settings, mode="flat")
        stats.chunks_created = len(chunks)
        stats.duration_ms = (time.perf_counter() - started) * 1000
        logger.info("ingest_complete", mode="flat", **stats.__dict__)
        return chunks, stats
