# Chroma DB Overview (2024)

Chroma is an open-source embedding database designed for AI applications.
It stores embeddings, documents, and metadata together, with persistence and
collection APIs that simplify application development.

## Strengths

- Persistent collections with a simple developer experience
- First-class metadata filtering for constrained retrieval
- Convenient local prototyping path that can grow into shared services

## Comparison with FAISS

FAISS optimizes for raw vector search speed. Chroma optimizes for productized
embedding storage with metadata and persistence ergonomics.
In production RAG, teams often prototype on Chroma and/or keep FAISS for
ultra-low-latency in-process retrieval hot paths.

## Practical guidance

If your queries frequently include filters such as category, tenant, or year,
Chroma's metadata filtering reduces the need for post-hoc filtering of FAISS hits.
If you need the absolute fastest local ANN scan, benchmark FAISS carefully.
