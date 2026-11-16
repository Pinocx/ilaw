"""Typed request and response contracts for the lexical RAG MVP."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class QueryUnderstanding(BaseModel):
    """Deterministic query signals that do not replace legal interpretation."""

    model_config = ConfigDict(extra="forbid")

    original_query: str
    explicit_article_references: list[str] = Field(default_factory=list)
    explicit_legal_identifiers: list[str] = Field(default_factory=list)
    explicit_date_mentions: list[str] = Field(default_factory=list)
    jurisdiction: str | None = None
    analysis_type: str = "legal_research"
    interpretation_status: Literal["deterministic_signals_only"] = "deterministic_signals_only"


class LegalSource(BaseModel):
    """Traceable document and chunk reference returned by retrieval."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    document_id: str
    filename: str | None = None
    document_title: str | None = None
    document_type: str | None = None
    classification_status: str
    chunk_id: str
    structure_type: str
    structure_reference: str | None = None
    parent_reference: str | None = None
    hierarchy_path: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    source_unit_ids: list[str] = Field(default_factory=list)
    legal_status: str = "unknown"
    legal_verification_status: str = "not_verified"
    content_trust: Literal["untrusted_source_content"] = "untrusted_source_content"
    retrieval_score: float
    retrieval_backend: str
    text: str
    relationships: list[dict[str, Any]] = Field(default_factory=list)


class LegalContextResponse(BaseModel):
    """Structured context ready for a downstream model or application."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    original_query: str
    query_understanding: QueryUnderstanding
    retrieval_method: str
    filters: dict[str, Any]
    sources: list[LegalSource]
    selected_source_count: int
    omitted_source_count: int
    context_characters: int
    context_char_budget: int
    insufficient_evidence: bool
    source_content_policy: str
    limitations: list[str]
