"""Settings validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hybrid_rag.config.settings import Settings


def test_overlap_must_be_lt_chunk_size():
    with pytest.raises(ValidationError):
        Settings(
            openai_api_key="sk-test",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_settings_defaults():
    s = Settings(openai_api_key="sk-test")
    assert s.top_k == 4
    assert s.default_vector_store.value == "faiss"
