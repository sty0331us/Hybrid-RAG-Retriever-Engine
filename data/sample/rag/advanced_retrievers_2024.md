# Advanced Retrievers for RAG (2024)

Retrieval-Augmented Generation (RAG) quality is dominated by retrieval quality.
A weak retriever cannot be fixed by a stronger generator.

## Vector store-backed retrievers

The baseline pattern embeds queries and documents into the same vector space,
then returns the top-k nearest chunks by cosine or L2 distance.
This approach is fast and simple, but brittle when users phrase questions poorly
or when relevant facts are split across chunks.

Production tip: always log retrieval latency, hit count, and source diversity.
Treat vector search as a measurable service, not a black box.

## Multi-query retrievers

A multi-query retriever uses an LLM to rewrite one user question into several
semantically diverse queries. Each variant is executed against the vector store,
and results are unioned / deduplicated.

Use multi-query when questions are ambiguous, multi-hop, or underspecified.
Cost: extra LLM calls and higher end-to-end latency.

## Self-querying retrievers

Self-query retrievers ask an LLM to translate natural language into:
1) a semantic search string, and 2) structured metadata filters
(e.g. category = "vector_db", year >= 2023).

They shine when your corpus has consistent metadata. They fail loudly when
metadata is missing or the filter language is mis-parsed.

## Parent document retrievers

Parent document retrieval indexes small child chunks for precise matching,
then returns larger parent documents to the generator.
This preserves local precision while restoring surrounding context needed for
faithful answers on long manuals, policies, and design docs.
