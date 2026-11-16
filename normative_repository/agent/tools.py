"""Typed allowlisted tool registry for legal research operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..config import CANDIDATE_INDEX_DIR
from ..llm.models import ToolDefinition
from ..rag.engine import (
    retrieve_legal_context,
    retrieve_legal_history,
    retrieve_related_provisions,
    search_legal_articles,
    search_legal_documents,
    verify_citation,
)
from ..rag.hybrid import HybridRetriever


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=8, ge=1, le=20)
    reviewed_only: bool = False


class ClassifiedSearchInput(SearchInput):
    classification: str | None = Field(default=None, max_length=64)


class ArticleLookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    article_reference: str = Field(min_length=1, max_length=100)
    query: str = Field(default="", max_length=1900)
    limit: int = Field(default=8, ge=1, le=20)
    reviewed_only: bool = False


class ContextInput(SearchInput):
    context_char_budget: int = Field(default=12000, ge=1, le=50000)
    classification: str | None = Field(default=None, max_length=64)
    structure_type: str | None = Field(default=None, max_length=64)


class CitationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(min_length=1, max_length=256)
    chunk_id: str = Field(min_length=1, max_length=256)
    quote: str | None = Field(default=None, max_length=12000)


class HistoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(min_length=1, max_length=256)


class RelationshipInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1, max_length=600)


class CompareInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    first_query: str = Field(min_length=1, max_length=2000)
    second_query: str = Field(min_length=1, max_length=2000)
    limit_per_query: int = Field(default=5, ge=1, le=10)


class DocumentHierarchyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=10)


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[..., dict[str, Any]]


class ToolExecutionError(ValueError):
    """Raised when a tool is not allowlisted or its arguments are invalid."""


class LegalToolRegistry:
    """Execute only registered, schema-validated local repository operations."""

    def __init__(
        self,
        *,
        index_dir: Path = CANDIDATE_INDEX_DIR,
        max_executions: int = 25,
        hybrid_retriever: HybridRetriever | None = None,
    ) -> None:
        if not 1 <= max_executions <= 100:
            raise ValueError("max_executions must be between 1 and 100.")
        self.index_dir = index_dir
        self.max_executions = max_executions
        self.hybrid_retriever = hybrid_retriever
        self._tools: dict[str, AgentTool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._register(AgentTool(
            "search_legal_documents", "Search candidate legal-document chunks.", ClassifiedSearchInput,
            lambda **data: search_legal_documents(index_dir=self.index_dir, **data),
        ))
        self._register(AgentTool(
            "search_legal_articles", "Search candidate chunks labeled as legal articles.", SearchInput,
            lambda **data: search_legal_articles(index_dir=self.index_dir, **data),
        ))
        self._register(AgentTool(
            "search_legislation", "Search candidate chunks belonging to legislation.", SearchInput,
            lambda query, limit=8, reviewed_only=False: search_legal_documents(
                index_dir=self.index_dir, query=query, limit=limit, classification="legislation", reviewed_only=reviewed_only,
            ),
        ))
        self._register(AgentTool(
            "search_case_law", "Search candidate chunks classified as jurisprudence.", SearchInput,
            lambda query, limit=8, reviewed_only=False: search_legal_documents(
                index_dir=self.index_dir, query=query, limit=limit, classification="jurisprudence", reviewed_only=reviewed_only,
            ),
        ))
        self._register(AgentTool(
            "get_legal_article", "Retrieve candidate article chunks for an explicit article reference.", ArticleLookupInput,
            lambda article_reference, query="", limit=8, reviewed_only=False: search_legal_articles(
                index_dir=self.index_dir, query=f"{article_reference} {query}".strip(), limit=limit, reviewed_only=reviewed_only,
            ),
        ))
        self._register(AgentTool(
            "retrieve_legal_context", "Retrieve complete traceable source chunks for a legal query.", ContextInput,
            lambda **data: retrieve_legal_context(index_dir=self.index_dir, hybrid_retriever=self.hybrid_retriever, **data),
        ))
        self._register(AgentTool(
            "verify_citation", "Verify source identity and optional exact quote in the candidate index.", CitationInput,
            lambda **data: verify_citation(index_dir=self.index_dir, **data),
        ))
        self._register(AgentTool(
            "retrieve_legal_history", "Return legal history when a reviewed version index is available.", HistoryInput,
            lambda document_id: retrieve_legal_history(document_id=document_id),
        ))
        self._register(AgentTool(
            "find_related_provisions", "Return legal relationships when a reviewed relationship index is available.", RelationshipInput,
            lambda source_id: retrieve_related_provisions(source_id=source_id),
        ))
        self._register(AgentTool(
            "get_document_hierarchy", "Retrieve matched hierarchy paths from candidate source chunks.", DocumentHierarchyInput,
            lambda query, limit=5: {
                "query": query,
                "hierarchy": [
                    {key: source.get(key) for key in ("source_id", "document_title", "structure_reference", "parent_reference", "hierarchy_path", "source_pages")}
                    for source in retrieve_legal_context(index_dir=self.index_dir, query=query, limit=limit, hybrid_retriever=self.hybrid_retriever).get("sources", [])
                ],
                "status": "candidate_hierarchy",
            },
        ))
        self._register(AgentTool(
            "check_legal_validity", "Report available validity metadata without inferring legal force.", CitationInput,
            lambda document_id, chunk_id, quote=None: {
                **verify_citation(index_dir=self.index_dir, document_id=document_id, chunk_id=chunk_id, quote=quote),
                "legal_validity_verified": False,
                "legal_validity_status": "unknown",
            },
        ))
        self._register(AgentTool(
            "compare_legal_provisions", "Retrieve evidence for two queries side by side without asserting a legal comparison.", CompareInput,
            lambda first_query, second_query, limit_per_query=5: {
                "first": retrieve_legal_context(index_dir=self.index_dir, query=first_query, limit=limit_per_query, hybrid_retriever=self.hybrid_retriever),
                "second": retrieve_legal_context(index_dir=self.index_dir, query=second_query, limit=limit_per_query, hybrid_retriever=self.hybrid_retriever),
                "analysis_status": "evidence_only",
            },
        ))

    def _register(self, tool: AgentTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def definitions(self) -> list[dict[str, Any]]:
        """Return JSON-schema tool declarations suitable for LLM proposals."""
        return [{
            "name": tool.name, "description": tool.description,
            "parameters": tool.input_model.model_json_schema(),
        } for tool in self._tools.values()]

    def llm_tool_definitions(self) -> list[ToolDefinition]:
        """Return validated tool declarations for providers that support tool calls."""
        return [ToolDefinition(**definition) for definition in self.definitions()]

    def execute_proposal(self, call_id: str, name: str, arguments: dict[str, Any], *, execution_count: int) -> dict[str, Any]:
        """Validate and execute one model-proposed call against the allowlist."""
        if not call_id or len(call_id) > 256:
            raise ToolExecutionError("Tool call identifier is invalid.")
        result = self.execute(name, arguments, execution_count=execution_count)
        return {"tool_call_id": call_id, "tool_name": name, "result": result}

    def execute(self, name: str, arguments: dict[str, Any], *, execution_count: int = 0) -> dict[str, Any]:
        if execution_count >= self.max_executions:
            raise ToolExecutionError("Maximum tool execution count exceeded.")
        tool = self._tools.get(name)
        if tool is None:
            raise ToolExecutionError("Requested tool is not allowlisted.")
        try:
            validated = tool.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError("Tool arguments do not match the registered schema.") from exc
        try:
            return tool.handler(**validated.model_dump())
        except (ValueError, OSError) as exc:
            raise ToolExecutionError("Tool execution failed safely.") from exc
