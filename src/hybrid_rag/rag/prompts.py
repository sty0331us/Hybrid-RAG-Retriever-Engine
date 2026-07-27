"""Prompt templates for grounded RAG generation."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM_PROMPT = """You are a precise technical assistant for a production RAG knowledge base.

Rules:
- Answer ONLY using the provided context chunks.
- If the context is insufficient, say you do not have enough information.
- Cite sources inline using [source:N] where N is the chunk rank.
- Be concise, factual, and production-engineering oriented.
- Never invent APIs, metrics, or configuration keys not present in context.
"""

RAG_HUMAN_PROMPT = """Context:
{context}

Question: {question}

Answer:"""


def build_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_HUMAN_PROMPT),
        ]
    )


def format_context(chunks: list) -> str:
    parts: list[str] = []
    for chunk in chunks:
        source = chunk.metadata.get("title") or chunk.metadata.get("source") or "unknown"
        category = chunk.metadata.get("category", "general")
        header = f"[source:{chunk.rank}] ({category} | {source})"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n---\n\n".join(parts) if parts else "No context retrieved."
