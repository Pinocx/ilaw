"""Normative Repository — bulk legal-document preparation pipeline.

A declarative pipeline for structuring, classifying, converting, and indexing
legally-relevant documents (codes, statutes, jurisprudence, doctrine) extracted
from manually downloaded collections into Markdown and JSONL for future STAIR
or RAG indexing.
"""

from normative_repository.segmentation import segment_document

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "segment_document",
]
