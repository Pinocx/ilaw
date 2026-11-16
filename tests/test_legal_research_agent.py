"""Tests for bounded evidence-first research orchestration."""

from __future__ import annotations

import unittest
from pathlib import Path

from normative_repository.agent import LegalResearchAgent


class LegalResearchAgentTests(unittest.TestCase):
    def test_retrieves_verifies_and_does_not_assert_legal_validity(self) -> None:
        calls = []

        def retrieve(query: str, **kwargs):
            calls.append((query, kwargs))
            return {
                "sources": [{
                    "source_id": "doc-1::chunk-1", "document_id": "doc-1",
                    "chunk_id": "chunk-1", "filename": "law.md", "source_pages": [2],
                }],
                "insufficient_evidence": False, "limitations": [],
            }

        def verify(**kwargs):
            return {"identity_verified": True, "verified": True}

        agent = LegalResearchAgent(index_dir=Path("index"), retrieve=retrieve, verify=verify)
        result = agent.run("What does this law say?")
        self.assertEqual(result.status, "evidence_retrieved")
        self.assertTrue(result.citations[0].traceability_verified)
        self.assertFalse(result.citations[0].legal_validity_verified)
        self.assertEqual(calls[0][1]["limit"], 8)
        self.assertTrue(any(event.event_type == "evidence_assessed" for event in result.events))

    def test_reports_insufficient_evidence_without_inventing_citations(self) -> None:
        agent = LegalResearchAgent(
            retrieve=lambda *args, **kwargs: {"sources": [], "insufficient_evidence": True, "limitations": []},
            verify=lambda **kwargs: self.fail("must not verify absent sources"),
        )
        result = agent.run("Unmatched question")
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.citations, [])

    def test_bounds_query_and_retrieval_limit(self) -> None:
        agent = LegalResearchAgent(retrieve=lambda *args, **kwargs: {})
        with self.assertRaises(ValueError):
            agent.run(" ")
        with self.assertRaises(ValueError):
            agent.run("question", limit=21)

    def test_deterministic_task_classifier_covers_legal_query_types(self) -> None:
        classify = LegalResearchAgent.classify_task
        self.assertEqual(classify("Busca jurisprudencia sobre contratos"), "jurisprudence_research")
        self.assertEqual(classify("¿Qué dice el artículo 10?"), "legal_article_lookup")
        self.assertEqual(classify("¿La norma está vigente?"), "legal_validity_analysis")
        self.assertEqual(classify("Compara ambas normas"), "legal_comparison")


if __name__ == "__main__":
    unittest.main()
