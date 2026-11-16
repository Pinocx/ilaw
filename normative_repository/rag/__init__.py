"""Provider-independent legal retrieval and context assembly."""

from .engine import (
    get_retrieval_sources,
    retrieve_legal_context,
    retrieve_legal_history,
    retrieve_related_provisions,
    search_legal_articles,
    search_legal_documents,
    verify_citation,
)
from .evaluation import evaluate_retrieval_dataset
from .hybrid import (
    HierarchyRetriever,
    HybridRetriever,
    OllamaEmbedder,
    PostgresPgVectorRetriever,
    SemanticRetriever,
    TextEmbedder,
)

__all__ = [
    "get_retrieval_sources",
    "evaluate_retrieval_dataset",
    "retrieve_legal_context",
    "retrieve_legal_history",
    "retrieve_related_provisions",
    "search_legal_articles",
    "search_legal_documents",
    "verify_citation",
    "HybridRetriever",
    "OllamaEmbedder",
    "PostgresPgVectorRetriever",
    "HierarchyRetriever",
    "SemanticRetriever",
    "TextEmbedder",
]
