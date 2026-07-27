# FAISS for Semantic Search (2023)

FAISS (Facebook AI Similarity Search) is a library for efficient similarity
search and clustering of dense vectors. It is widely used for in-process
vector retrieval in RAG systems.

## Strengths

- Excellent raw search throughput on CPU and GPU
- Fine-grained index types (Flat, IVF, HNSW, PQ) for recall/speed trade-offs
- Ideal for embedded services and low-latency local retrieval

## Operational considerations

FAISS does not provide a full database product out of the box.
You own persistence format, metadata filtering, multi-tenancy, and replication.
In LangChain, FAISS indexes are commonly saved with `save_local` / `load_local`.

## When FAISS wins

Choose FAISS when you need maximum similarity-search performance inside a
single service process and can manage persistence yourself.
