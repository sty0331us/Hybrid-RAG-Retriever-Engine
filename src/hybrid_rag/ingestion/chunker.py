"""Text splitting / hierarchical chunking."""

from __future__ import annotations

import uuid
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_rag.config.settings import Settings
from hybrid_rag.core.logging import get_logger

logger = get_logger(__name__)


def build_splitter(
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(
    documents: list[Document],
    settings: Settings,
    *,
    mode: Literal["flat", "parent_child"] = "flat",
) -> list[Document]:
    """
    Chunk documents.

    - flat: single-level chunks (vector / multi-query / self-query)
    - parent_child: small child chunks for retrieval + parent_id linkage
    """
    if mode == "flat":
        splitter = build_splitter(settings.chunk_size, settings.chunk_overlap)
        chunks = splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            doc_id = chunk.metadata.get("doc_id") or str(uuid.uuid4())[:16]
            chunk.metadata.setdefault("doc_id", doc_id)
        logger.info("chunks_created", mode=mode, count=len(chunks))
        return chunks

    # Parent / child hierarchy for ParentDocumentRetriever
    parent_splitter = build_splitter(settings.parent_chunk_size, settings.chunk_overlap)
    child_splitter = build_splitter(settings.child_chunk_size, max(20, settings.chunk_overlap // 2))

    parents = parent_splitter.split_documents(documents)
    children: list[Document] = []
    for parent in parents:
        parent_id = parent.metadata.get("doc_id") or str(uuid.uuid4())
        parent.metadata["doc_id"] = parent_id
        parent.metadata["is_parent"] = True
        child_docs = child_splitter.split_documents([parent])
        for idx, child in enumerate(child_docs):
            child.metadata = {
                **parent.metadata,
                **child.metadata,
                "parent_id": parent_id,
                "is_parent": False,
                "chunk_index": idx,
            }
            children.append(child)

    logger.info(
        "chunks_created",
        mode=mode,
        parents=len(parents),
        children=len(children),
    )
    # Return children for indexing; parents are tracked via metadata / docstore
    # Callers that need parents should use chunk_parent_child()
    return children


def chunk_parent_child(
    documents: list[Document],
    settings: Settings,
) -> tuple[list[Document], list[Document]]:
    """Return (parents, children) for Parent Document retrieval."""
    parent_splitter = build_splitter(settings.parent_chunk_size, settings.chunk_overlap)
    child_splitter = build_splitter(settings.child_chunk_size, max(20, settings.chunk_overlap // 2))

    parents = parent_splitter.split_documents(documents)
    children: list[Document] = []
    for parent in parents:
        parent_id = str(uuid.uuid4())
        parent.metadata["doc_id"] = parent_id
        parent.metadata["is_parent"] = True
        for idx, child in enumerate(child_splitter.split_documents([parent])):
            child.metadata = {
                **parent.metadata,
                **child.metadata,
                "parent_id": parent_id,
                "doc_id": f"{parent_id}-{idx}",
                "is_parent": False,
                "chunk_index": idx,
            }
            children.append(child)
    return parents, children
