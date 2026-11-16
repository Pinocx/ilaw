"""Tests for provider-independent legal retrieval contracts."""

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from normative_repository.config import PROJECT_ROOT
from normative_repository.rag.engine import (
    get_retrieval_sources,
    retrieve_legal_context,
    retrieve_legal_history,
    retrieve_related_provisions,
    understand_query,
    verify_citation,
)


class RagEngineTests(unittest.TestCase):
    def setUp(self):
        self.index_dir = PROJECT_ROOT / "data" / "review" / f".rag-test-{__import__('uuid').uuid4().hex}"
        self.index_dir.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.index_dir)
        self.document_id = "doc-1"
        self.chunk_id = "doc-1:chunk-000001"
        (self.index_dir / "documents.jsonl").write_text(
            json.dumps({
                "doc_id": self.document_id,
                "filename": "Constitution.pdf",
                "document_metadata": {
                    "document_title": None,
                    "document_type": "legislation",
                    "legal_status": "unknown",
                },
            }) + "\n",
            encoding="utf-8",
        )
        self.chunk = {
            "doc_id": self.document_id,
            "chunk_id": self.chunk_id,
            "structure_type": "article",
            "structure_reference": "Artículo 1",
            "parent_reference": "Title I",
            "hierarchy_path": ["Title I", "Artículo 1"],
            "source_pages": [4],
            "source_unit_ids": ["doc-1:page-000004"],
            "text": "La defensa de la persona humana es el fin supremo de la sociedad y del Estado.",
            "score": 100,
            "reviewed_classification": "legislation",
            "classification_status": "reviewed",
            "legal_verification_status": "not_verified",
        }
        (self.index_dir / "chunks.jsonl").write_text(
            json.dumps(self.chunk, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def test_query_understanding_preserves_original_and_only_extracts_explicit_signals(self):
        query = "¿Qué dispone el artículo 1 de la Ley N° 123-2020 en Perú desde 1 de enero de 2024?"
        result = understand_query(query)

        self.assertEqual(result.original_query, query)
        self.assertEqual(result.explicit_article_references, ["Artículo 1"])
        self.assertEqual(result.explicit_legal_identifiers, ["Ley N° 123-2020"])
        self.assertEqual(result.explicit_date_mentions, ["1 de enero de 2024"])
        self.assertEqual(result.jurisdiction, "PE")
        self.assertEqual(result.interpretation_status, "deterministic_signals_only")

    def test_context_preserves_source_and_marks_source_text_untrusted(self):
        with patch("normative_repository.rag.engine.search_chunks", return_value={
            "search_backend": "test",
            "filters": {},
            "results": [self.chunk],
        }):
            result = retrieve_legal_context(
                "defensa persona",
                index_dir=self.index_dir,
                context_char_budget=1000,
            )

        self.assertEqual(result["original_query"], "defensa persona")
        self.assertEqual(result["selected_source_count"], 1)
        source = result["sources"][0]
        self.assertEqual(source["document_title"], "Constitution.pdf")
        self.assertEqual(source["source_pages"], [4])
        self.assertEqual(source["source_unit_ids"], ["doc-1:page-000004"])
        self.assertEqual(source["legal_status"], "unknown")
        self.assertEqual(source["content_trust"], "untrusted_source_content")
        self.assertIn("never executable instructions", result["source_content_policy"])
        self.assertEqual(get_retrieval_sources(result)[0]["chunk_id"], self.chunk_id)

    def test_context_never_truncates_chunks_to_fit_budget(self):
        large_chunk = {**self.chunk, "text": "x" * 101}
        with patch("normative_repository.rag.engine.search_chunks", return_value={
            "search_backend": "test",
            "filters": {},
            "results": [large_chunk],
        }):
            result = retrieve_legal_context(
                "defensa persona",
                index_dir=self.index_dir,
                context_char_budget=100,
            )

        self.assertEqual(result["sources"], [])
        self.assertEqual(result["omitted_source_count"], 1)
        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["context_characters"], 0)

    def test_context_can_use_hybrid_ranked_results_when_adapter_is_configured(self):
        class FakeHybridRetriever:
            def retrieve(self, query, *, limit, **filters):
                self.called_with = (query, limit, filters)
                return {
                    "search_backend": "hybrid_rrf",
                    "retrieval_methods": ["lexical", "semantic"],
                    "filters": filters,
                    "results": [{**self.chunk, "score": 0.02, "retrieval_methods": ["semantic"]}],
                }

            chunk = None

        retriever = FakeHybridRetriever()
        retriever.chunk = self.chunk
        with patch("normative_repository.rag.engine._load_documents", return_value={
            self.document_id: {"filename": "Constitution.pdf"},
        }):
            context = retrieve_legal_context(
                "defensa persona", index_dir=self.index_dir, hybrid_retriever=retriever,
            )
        self.assertEqual(context["retrieval_method"], "hybrid_rrf")
        self.assertEqual(context["sources"][0]["retrieval_score"], 0.02)
        self.assertNotIn("Semantic similarity is not configured for this request.", context["limitations"])

    def test_citation_verification_checks_identity_and_exact_quote(self):
        valid = verify_citation(
            index_dir=self.index_dir,
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            quote="fin supremo de la sociedad",
        )
        invalid = verify_citation(
            index_dir=self.index_dir,
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            quote="the supreme purpose",
        )

        self.assertTrue(valid["verified"])
        self.assertEqual(valid["source_pages"], [4])
        self.assertEqual(valid["legal_status"], "unknown")
        self.assertFalse(invalid["verified"])
        self.assertEqual(invalid["reason"], "quote_not_found_in_chunk")

    def test_history_and_relationships_are_explicitly_unavailable(self):
        self.assertEqual(
            retrieve_legal_history(document_id=self.document_id)["status"],
            "unavailable",
        )
        self.assertEqual(
            retrieve_related_provisions(source_id=self.chunk_id)["status"],
            "unavailable",
        )

    def test_query_and_context_limits_are_enforced(self):
        with self.assertRaises(ValueError):
            understand_query("  ")
        with patch("normative_repository.rag.engine.search_chunks", return_value={
            "search_backend": "test", "filters": {}, "results": [],
        }):
            with self.assertRaises(ValueError):
                retrieve_legal_context("valid query", index_dir=self.index_dir, context_char_budget=0)


if __name__ == "__main__":
    unittest.main()
