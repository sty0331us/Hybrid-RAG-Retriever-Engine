# Hybrid RAG Retriever Engine

Production-oriented system for **comparing and operating advanced RAG retrievers**.
Built to demonstrate real engineering judgment — modular architecture, measurable
retrieval trade-offs, and an operator UI — the kind of artifact hiring managers
and recruiters can actually review.

## What this proves

| Course concept | Production implementation |
|---|---|
| Vector store-backed retrievers | `VectorStoreRetrieverStrategy` over FAISS / Chroma |
| Multi-query retrievers | LLM query expansion + result union |
| Self-querying retrievers | NL → semantic query + metadata filters |
| Parent document retrievers | Child-chunk match → parent-context return |
| FAISS vs Chroma | Dedicated store managers + benchmark tab |
| End-to-end RAG + Gradio UI | `RAGEngine` + operator console |

## Architecture

```
                         ┌─────────────────────────────────────┐
                         │           Query / Operator          │
                         │   CLI · Gradio UI · Docker service  │
                         └──────────────────┬──────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              RAGEngine façade                                │
│                     ingest → index → retrieve → generate                     │
└───────────────┬───────────────────────────────────────────────┬──────────────┘
                │                                               │
                ▼                                               ▼
┌───────────────────────────────┐               ┌──────────────────────────────┐
│         Ingestion             │               │     Vector Store Backends    │
│  load (.txt/.md/.pdf)         │──────────────▶│                              │
│  normalize metadata           │               │  ┌────────┐    ┌──────────┐  │
│  (source, category, year,     │               │  │ FAISS  │    │  Chroma  │  │
│   title, doc_id)              │               │  │ flat   │    │  flat    │  │
│  flat chunking                │               │  │ parent │    │  parent  │  │
│  parent/child hierarchy       │               │  └────────┘    └──────────┘  │
└───────────────────────────────┘               └───────────────┬──────────────┘
                                                                │
                                                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Retriever Strategies (pluggable)                     │
│                                                                              │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────┐  │
│  │   vector_store     │  │    multi_query     │  │      self_query       │  │
│  │  dense similarity  │  │  LLM rewrites Q →  │  │  LLM → semantic query │  │
│  │  top-k over chunks │  │  Q1..Qn, union hits│  │  + metadata filters   │  │
│  │  baseline / fast   │  │  ambiguous Qs      │  │  (category, year, …)  │  │
│  └────────────────────┘  └────────────────────┘  └────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        parent_document                                 │  │
│  │   match small child chunks  →  return larger parent context            │  │
│  │   best for long manuals / policies where local match needs surround    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ RetrievedChunk[]
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Grounded Generation                                  │
│   context formatting + citations [source:N]  →  ChatOpenAI (retry/backoff)   │
│   refusal when context insufficient                                          │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ RAGAnswer
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Evaluation / Metrics (RetrieverBenchmark)                 │
│                                                                              │
│   Per-run metrics              Comparison outputs                            │
│   • retrieval_latency_ms       • compare_retrievers()  — all 4 strategies    │
│   • generation_latency_ms      • compare_vector_stores() — FAISS vs Chroma   │
│   • total_latency_ms                                                         │
│   • num_chunks                 Ranking / summary                             │
│   • avg_score                  • fastest_strategy                            │
│   • unique_sources             • most_diverse_strategy                       │
│   • answer_preview             • StrategyBenchmark rows (table/JSON)         │
│                                                                              │
│   Surfaces: Gradio Compare tabs · CLI compare-* · CI-friendly JSON summary   │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Project layout

```
src/hybrid_rag/
  config/          # pydantic-settings
  core/            # logging, exceptions, domain types
  ingestion/       # loaders + chunkers
  stores/          # FAISS & Chroma managers
  retrievers/      # four strategy implementations
  rag/             # LLM clients + pipeline façade
  evaluation/      # latency / diversity benchmarks
  ui/              # Gradio console
  cli/             # Typer CLI
data/sample/       # curated knowledge base (category + year metadata)
tests/unit/        # no-network unit tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env               # add your OPENAI_API_KEY
```

## Usage

**Index the sample corpus (FAISS + Chroma, flat + parent indexes):**

```bash
hybrid-rag ingest data/sample
```

**Single RAG query:**

```bash
hybrid-rag ask "What are the trade-offs of parent document retrieval?" \
  --strategy parent_document --backend faiss
```

**Compare all retriever strategies:**

```bash
hybrid-rag compare-retrievers "Explain FAISS vs Chroma for production RAG"
```

**Compare vector stores:**

```bash
hybrid-rag compare-stores "How does metadata filtering help retrieval?" \
  --strategy self_query
```

**Launch Gradio UI:**

```bash
hybrid-rag ui
# open http://127.0.0.1:7860
```

## Docker

```bash
cp .env.example .env   # set OPENAI_API_KEY
docker compose up --build
```

## Retriever strategy cheat-sheet

| Strategy | Best for | Cost / latency |
|---|---|---|
| `vector_store` | Clear semantic questions | Lowest |
| `multi_query` | Ambiguous / multi-angle questions | Extra LLM rewrites |
| `self_query` | Filtered questions (`category`, `year`, …) | LLM parse + filter |
| `parent_document` | Long docs needing surrounding context | Indexing complexity |

### Example self-query prompts

- `Find 2024 documents in the vector_db category about Chroma`
- `Security guidance from 2025 about prompt injection`

## Tests

```bash
OPENAI_API_KEY=sk-test pytest -q tests/unit
```

## Design choices (interview talking points)

1. **Strategy pattern** — retrievers share one interface; swapping is a config change.
2. **Isolated indexes** — flat chunks and parent/child indexes are separated per backend.
3. **Measurable comparisons** — latency, chunk count, and source diversity are first-class.
4. **Operational basics** — env-based secrets, structlog, retries, Docker, CI.
5. **Grounded generation** — prompts enforce citation + refusal when context is weak.

## License

MIT
