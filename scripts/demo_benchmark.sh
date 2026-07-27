#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy .env.example and set OPENAI_API_KEY"
  exit 1
fi

echo "==> Ingesting sample corpus"
hybrid-rag ingest data/sample --rebuild

echo "==> Comparing retriever strategies"
hybrid-rag compare-retrievers \
  "When should I choose multi-query over a plain vector store retriever?" \
  --backend faiss

echo "==> Comparing vector stores"
hybrid-rag compare-stores \
  "What are FAISS strengths for semantic search?" \
  --strategy vector_store

echo "Done."
