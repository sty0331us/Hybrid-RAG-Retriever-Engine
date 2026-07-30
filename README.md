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
| Hybrid dense + lexical | `EnsembleHybridRetrieverStrategy` with RRF fusion |
| FAISS vs Chroma | Dedicated store managers + benchmark tab |
| End-to-end RAG + Gradio UI | `RAGEngine` + operator console |
| REST API + metrics | FastAPI `/ask` · Prometheus `/metrics` |

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
│  ┌──────────────────────────────┐  ┌──────────────────────────────────────┐  │
│  │      parent_document         │  │             ensemble                 │  │
│  │  match child → return parent │  │  dense + lexical RRF fusion          │  │
│  │  long manuals / policies     │  │  keyword + semantic hybrid           │  │
│  └──────────────────────────────┘  └──────────────────────────────────────┘  │
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
│   • retrieval_latency_ms       • compare_retrievers()  — all strategies      │
│   • generation_latency_ms      • compare_vector_stores() — FAISS vs Chroma   │
│   • total_latency_ms           • eval-quality — precision@k / recall@k / MRR │
│   • num_chunks                                                               │
│   • avg_score                  Ranking / summary                             │
│   • unique_sources             • fastest_strategy                            │
│   • answer_preview             • most_diverse_strategy                       │
│                                • StrategyBenchmark rows (table/JSON)         │
│                                                                              │
│   Surfaces: Gradio · CLI · FastAPI /metrics · CI-friendly JSON summary       │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Project layout

```
src/hybrid_rag/
  api/             # FastAPI REST service
  config/          # pydantic-settings
  core/            # logging, exceptions, domain types, TTL cache
  ingestion/       # loaders + chunkers
  stores/          # FAISS & Chroma managers
  retrievers/      # strategy implementations (incl. ensemble hybrid)
  rag/             # LLM clients + pipeline façade
  evaluation/      # latency benchmarks + precision@k / MRR
  observability/   # Prometheus metrics
  ui/              # Gradio console
  cli/             # Typer CLI
data/sample/       # curated knowledge base + golden_queries.json
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

**Hybrid dense + lexical retrieval:**

```bash
hybrid-rag ask "FAISS similarity search trade-offs" --strategy ensemble
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

**Offline retrieval quality (precision@k / MRR):**

```bash
hybrid-rag eval-quality data/sample/golden_queries.json --strategy ensemble
```

**Launch Gradio UI:**

```bash
hybrid-rag ui
# open http://127.0.0.1:7860
```

**Launch FastAPI REST service:**

```bash
pip install -e ".[api]"   # if you only installed the base package
hybrid-rag serve
# docs: http://127.0.0.1:8000/docs
# health: GET /health
# ask: POST /ask
# metrics: GET /metrics
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
| `ensemble` | Keyword + semantic hybrid queries | Slightly above vector |

### Example self-query prompts

- `Find 2024 documents in the vector_db category about Chroma`
- `Security guidance from 2025 about prompt injection`

## Tests

```bash
OPENAI_API_KEY=sk-test pytest -q tests/unit
```

## Design choices (interview talking points)

1. **Strategy pattern** — retrievers share one interface; swapping is a config change.
2. **True hybrid retrieval** — ensemble fuses dense + lexical rankings with RRF.
3. **Isolated indexes** — flat chunks and parent/child indexes are separated per backend.
4. **Measurable comparisons** — latency, diversity, and precision@k / MRR are first-class.
5. **Operational basics** — env secrets, structlog, Prometheus, TTL cache, Docker, CI.
6. **Multiple surfaces** — CLI, Gradio, and FastAPI for demos vs integration.
7. **Grounded generation** — prompts enforce citation + refusal when context is weak.

## License

MIT
