"""Tests for deterministic candidate-index text search."""

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from normative_repository.config import PROJECT_ROOT
from normative_repository.search import search_chunks
from normative_repository.text_index import build_text_search_index


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.index_dir = PROJECT_ROOT / "data" / "review" / f".search-test-{uuid4().hex}"
        self.index_dir.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.index_dir)
        (self.index_dir / "documents.jsonl").write_text(
            "\n".join([
                json.dumps({"doc_id": "reviewed-doc", "filename": "Código Penal.pdf"}),
                json.dumps({"doc_id": "pending-doc", "filename": "Manual jurídico.pdf"}),
            ]) + "\n",
            encoding="utf-8",
        )
        rows = [
            {
                "chunk_id": "reviewed-doc:chunk-1",
                "doc_id": "reviewed-doc",
                "structure_type": "article",
                "structure_reference": "Artículo 5",
                "parent_reference": "Título Preliminar",
                "hierarchy_path": ["Título Preliminar", "Artículo 5"],
                "source_pages": [4],
                "source_unit_ids": ["reviewed-doc:page-000004"],
                "text": "La acción penal prescribe según la ley.",
                "suggested_classification": "legislation",
                "reviewed_classification": "legislation",
                "classification_status": "reviewed",
                "legal_verification_status": "not_verified",
            },
            {
                "chunk_id": "pending-doc:chunk-1",
                "doc_id": "pending-doc",
                "structure_type": "paragraph",
                "structure_reference": None,
                "parent_reference": None,
                "hierarchy_path": [],
                "source_pages": [2],
                "source_unit_ids": ["pending-doc:page-000002"],
                "text": "La acción penal prescribe conforme a la ley.",
                "suggested_classification": "legal_manual",
                "reviewed_classification": None,
                "classification_status": "pending_review",
            },
        ]
        (self.index_dir / "chunks.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_all_terms_match_accent_insensitively_and_keep_provenance(self):
        result = search_chunks("accion penal", index_dir=self.index_dir)

        self.assertEqual(result["total_matches"], 2)
        hit = result["results"][0]
        self.assertEqual(hit["chunk_id"], "reviewed-doc:chunk-1")
        self.assertEqual(hit["source_pages"], [4])
        self.assertEqual(hit["source_unit_ids"], ["reviewed-doc:page-000004"])
        self.assertEqual(hit["filename"], "Código Penal.pdf")
        self.assertEqual(hit["matched_fields"], ["text"])

    def test_exact_phrase_and_review_filters(self):
        result = search_chunks(
            "acción penal prescribe según la ley",
            index_dir=self.index_dir,
            exact_phrase=True,
            reviewed_only=True,
            classification="legislation",
            structure_type="article",
        )

        self.assertEqual(result["total_matches"], 1)
        self.assertEqual(result["results"][0]["classification_status"], "reviewed")

    def test_document_title_queries_identify_the_matching_field(self):
        result = search_chunks(
            "codigo penal",
            index_dir=self.index_dir,
            exact_phrase=True,
        )

        self.assertEqual(result["total_matches"], 1)
        self.assertEqual(result["results"][0]["matched_fields"], ["document_title"])

    def test_result_limit_bounds_returned_hits_without_losing_total(self):
        result = search_chunks("acción penal", index_dir=self.index_dir, limit=1)

        self.assertEqual(result["total_matches"], 2)
        self.assertEqual(result["returned_matches"], 1)
        self.assertEqual(len(result["results"]), 1)

    def test_invalid_queries_and_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            search_chunks("   ", index_dir=self.index_dir)
        with self.assertRaises(ValueError):
            search_chunks("law", index_dir=self.index_dir, limit=0)

    def test_persistent_fts_index_preserves_substring_and_phrase_results(self):
        build_text_search_index(self.index_dir)

        substring_result = search_chunks("enal prescribe", index_dir=self.index_dir)
        phrase_result = search_chunks(
            "acción penal prescribe según la ley",
            index_dir=self.index_dir,
            exact_phrase=True,
        )

        self.assertEqual(substring_result["search_backend"], "sqlite_fts5_trigram")
        self.assertEqual(substring_result["total_matches"], 2)
        self.assertEqual(phrase_result["total_matches"], 1)
        self.assertEqual(phrase_result["results"][0]["chunk_id"], "reviewed-doc:chunk-1")

    def test_short_terms_and_stale_index_fall_back_to_jsonl(self):
        build_text_search_index(self.index_dir)
        short_term_result = search_chunks("la ley", index_dir=self.index_dir)
        self.assertEqual(short_term_result["search_backend"], "jsonl_scan")

        with (self.index_dir / "chunks.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "chunk_id": "new-doc:chunk-1",
                "doc_id": "pending-doc",
                "structure_type": "paragraph",
                "text": "Acción penal adicional.",
            }) + "\n")
        stale_result = search_chunks("accion penal", index_dir=self.index_dir)

        self.assertEqual(stale_result["search_backend"], "jsonl_scan")
        self.assertEqual(stale_result["search_index_fallback_reason"], "search_index_stale")
        self.assertEqual(stale_result["total_matches"], 3)


if __name__ == "__main__":
    unittest.main()
