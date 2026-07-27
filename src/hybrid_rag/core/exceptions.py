"""Domain exception hierarchy for predictable error handling."""

from __future__ import annotations


class HybridRAGError(Exception):
    """Base exception for all Hybrid RAG errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(HybridRAGError):
    """Invalid or missing configuration."""


class IngestionError(HybridRAGError):
    """Document loading or chunking failed."""


class StoreError(HybridRAGError):
    """Vector store operation failed."""


class RetrieverError(HybridRAGError):
    """Retriever construction or query failed."""


class RAGPipelineError(HybridRAGError):
    """End-to-end RAG generation failed."""


class EvaluationError(HybridRAGError):
    """Benchmark / metrics computation failed."""
