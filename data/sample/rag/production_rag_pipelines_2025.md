# Designing Production RAG Pipelines (2025)

A production RAG system is more than "embed docs + call LLM".

## Ingestion

- Normalize metadata at ingest time (source, title, category, year, doc_id).
- Choose chunk sizes intentionally; overlap should be smaller than chunk size.
- Keep a stable document identity so re-indexing is idempotent.

## Indexing

Separate indexes when retrieval strategies need different chunking schemes.
For example, flat chunk indexes for vector / multi-query / self-query, and a
dedicated child-chunk index for parent-document retrieval.

## Generation

Ground the model with explicit citation rules. Prefer extractive faithfulness
over fluent hallucination. Retry transient LLM failures with exponential backoff.

## Evaluation

Compare retrievers on the same query set using:
- latency (p50 / p95)
- number of unique sources
- answer groundedness (human or LLM-as-judge)
- empty-result rate for filtered queries

Never promote a retriever strategy to production based on a single cherry-picked demo query.
