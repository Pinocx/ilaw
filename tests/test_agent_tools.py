"""Security and schema tests for the legal tool registry."""

from __future__ import annotations

import unittest
from pathlib import Path

from normative_repository.agent import LegalToolRegistry, ToolExecutionError


class LegalToolRegistryTests(unittest.TestCase):
    def test_registry_exposes_required_allowlisted_tools(self) -> None:
        registry = LegalToolRegistry(index_dir=Path("unused"))
        self.assertTrue({
            "search_legal_documents", "search_legal_articles", "retrieve_legal_context",
            "search_legislation", "search_case_law", "get_legal_article", "get_document_hierarchy",
            "check_legal_validity", "compare_legal_provisions", "verify_citation",
            "retrieve_legal_history", "find_related_provisions",
        }.issubset(registry.tool_names))
        self.assertEqual(len(registry.definitions()), len(registry.tool_names))
        self.assertEqual(len(registry.llm_tool_definitions()), len(registry.tool_names))

    def test_rejects_unregistered_tool_and_extra_arguments(self) -> None:
        registry = LegalToolRegistry()
        with self.assertRaises(ToolExecutionError):
            registry.execute("arbitrary_python", {})
        with self.assertRaises(ToolExecutionError):
            registry.execute("verify_citation", {"document_id": "d", "chunk_id": "c", "shell": "unsafe"})

    def test_enforces_execution_limit_and_input_bounds(self) -> None:
        registry = LegalToolRegistry(max_executions=1)
        with self.assertRaises(ToolExecutionError):
            registry.execute("verify_citation", {"document_id": "d", "chunk_id": "c"}, execution_count=1)
        with self.assertRaises(ToolExecutionError):
            registry.execute("search_legal_documents", {"query": "x", "limit": 100})

    def test_history_and_relationship_tools_disclose_unavailable_data(self) -> None:
        registry = LegalToolRegistry()
        self.assertEqual(
            registry.execute("retrieve_legal_history", {"document_id": "doc"})["status"],
            "unavailable",
        )

    def test_model_proposal_is_validated_and_tied_to_tool_call_id(self) -> None:
        registry = LegalToolRegistry()
        with self.assertRaises(ToolExecutionError):
            registry.execute_proposal("call-1", "run_shell", {}, execution_count=0)
        self.assertEqual(
            registry.execute("find_related_provisions", {"source_id": "doc::chunk"})["status"],
            "unavailable",
        )


if __name__ == "__main__":
    unittest.main()
