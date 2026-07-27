"""Ingestion loader tests against bundled sample corpus."""

from __future__ import annotations

from pathlib import Path

from hybrid_rag.ingestion.loader import load_directory, load_file

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample"


def test_load_sample_directory():
    docs = load_directory(SAMPLE)
    assert len(docs) >= 6
    categories = {d.metadata["category"] for d in docs}
    assert {"rag", "vector_db", "llm", "security"} <= categories
    assert all(d.metadata.get("doc_id") for d in docs)


def test_year_inferred_from_filename():
    path = SAMPLE / "vector_db" / "faiss_semantic_search_2023.md"
    docs = load_file(path)
    assert docs
    assert docs[0].metadata.get("year") == 2023
