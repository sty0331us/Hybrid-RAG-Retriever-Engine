"""Self-querying retriever: LLM translates NL → structured metadata filter + query."""

from __future__ import annotations

from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever

from hybrid_rag.core.logging import get_logger
from hybrid_rag.retrievers.base import BaseStrategyRetriever
from hybrid_rag.stores.base import BaseVectorStoreManager

logger = get_logger(__name__)

# Document attribute schema used by the query constructor.
# Keep these aligned with metadata written during ingestion.
DEFAULT_METADATA_FIELD_INFO: list[AttributeInfo] = [
    AttributeInfo(
        name="category",
        description="Topic category of the document, e.g. rag, vector_db, llm, security",
        type="string",
    ),
    AttributeInfo(
        name="title",
        description="Human-readable document title",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="Publication or document year as an integer",
        type="integer",
    ),
    AttributeInfo(
        name="source",
        description="Filesystem path or URI of the source document",
        type="string",
    ),
]

DOCUMENT_CONTENT_DESCRIPTION = (
    "Technical knowledge base covering RAG systems, vector databases, "
    "retrievers, LLMs, and production AI engineering practices."
)


class SelfQueryRetrieverStrategy(BaseStrategyRetriever):
    """
    Uses an LLM to extract semantic query + metadata filters from natural language.

    Best for: questions that constrain by category, year, source, etc.
    Trade-off: requires consistent metadata; filter misparse can return empty sets.
    """

    strategy_name = "self_query"

    def __init__(
        self,
        store: BaseVectorStoreManager,
        llm: BaseLanguageModel,
        *,
        top_k: int = 4,
        metadata_field_info: list[AttributeInfo] | None = None,
        document_contents: str = DOCUMENT_CONTENT_DESCRIPTION,
        enable_limit: bool = True,
    ) -> None:
        super().__init__(store.backend_name, top_k=top_k)
        self.store = store
        self.llm = llm
        self.metadata_field_info = metadata_field_info or DEFAULT_METADATA_FIELD_INFO
        self.document_contents = document_contents
        self.enable_limit = enable_limit

    def build(self) -> BaseRetriever:
        self._retriever = SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.store.store,
            document_contents=self.document_contents,
            metadata_field_info=self.metadata_field_info,
            enable_limit=self.enable_limit,
            search_kwargs={"k": self.top_k},
        )
        logger.info("self_query_retriever_ready", top_k=self.top_k)
        return self._retriever
