.PHONY: install test lint ui ingest demo serve docker-up

install:
	pip install -e ".[dev]"

test:
	OPENAI_API_KEY=$${OPENAI_API_KEY:-sk-test} pytest -q tests/unit

lint:
	ruff check src tests

ui:
	hybrid-rag ui

ingest:
	hybrid-rag ingest data/sample

demo:
	bash scripts/demo_benchmark.sh

serve:
	hybrid-rag serve

docker-up:
	docker compose up --build
