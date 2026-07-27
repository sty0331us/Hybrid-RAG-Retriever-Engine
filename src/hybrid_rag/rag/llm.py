"""LLM / embedding client factories with retry-aware defaults."""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from hybrid_rag.config.settings import Settings


def build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.embedding_model,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
