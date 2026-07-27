"""Document loading utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from hybrid_rag.core.exceptions import IngestionError
from hybrid_rag.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def _stable_doc_id(path: Path, content: str) -> str:
    digest = hashlib.sha256(f"{path.resolve()}::{content[:2048]}".encode()).hexdigest()
    return digest[:16]


def _infer_category(path: Path) -> str:
    # Prefer immediate parent folder under data/ as category
    parts = path.parts
    if "sample" in parts:
        idx = parts.index("sample")
        if idx + 1 < len(parts) - 1:
            return parts[idx + 1]
    return path.parent.name or "general"


def load_file(path: Path) -> list[Document]:
    """Load a single supported file into LangChain Documents."""
    if not path.exists():
        raise IngestionError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type: {suffix}",
            details={"supported": sorted(SUPPORTED_EXTENSIONS)},
        )

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            # txt + markdown: plain text load keeps dependency surface small
            loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Failed to load {path}: {exc}") from exc

    enriched: list[Document] = []
    for i, doc in enumerate(docs):
        content = doc.page_content.strip()
        if not content:
            continue
        meta = {
            **doc.metadata,
            "source": str(path),
            "doc_id": _stable_doc_id(path, content),
            "category": _infer_category(path),
            "title": path.stem.replace("_", " ").title(),
            "chunk_index": i,
        }
        # Best-effort year from filename like report_2023.txt
        for token in path.stem.split("_"):
            if token.isdigit() and len(token) == 4:
                meta["year"] = int(token)
                break
        enriched.append(Document(page_content=content, metadata=meta))

    logger.info("file_loaded", path=str(path), documents=len(enriched))
    return enriched


def load_directory(directory: Path, *, recursive: bool = True) -> list[Document]:
    """Load all supported documents from a directory."""
    if not directory.exists():
        raise IngestionError(f"Directory not found: {directory}")

    documents: list[Document] = []
    pattern = "**/*" if recursive else "*"
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                documents.extend(load_file(path))
            except IngestionError as exc:
                logger.warning("skip_file", path=str(path), error=str(exc))
    logger.info("directory_loaded", directory=str(directory), documents=len(documents))
    return documents


def load_paths(paths: Iterable[Path]) -> list[Document]:
    documents: list[Document] = []
    for path in paths:
        if path.is_dir():
            documents.extend(load_directory(path))
        else:
            documents.extend(load_file(path))
    return documents
