"""Smoke tests for the implemented normative repository modules."""

import unittest
from pathlib import Path

from normative_repository.cli import _resolve_index_output_dir
from normative_repository.config import CANDIDATE_INDEX_DIR, INDEX_DIR
from normative_repository.index import build_index_dataset
from normative_repository.inventory import _preserve_processing_metadata
from normative_repository.pipeline import validate_outputs
from normative_repository.segmentation import segment_document
from normative_repository.schemas import Chunk, ChunkMetadata


class CoreImportTests(unittest.TestCase):
    def test_core_modules_import(self):
        self.assertTrue(callable(segment_document))
        self.assertTrue(callable(build_index_dataset))
        self.assertTrue(callable(validate_outputs))

    def test_chunk_schema_supports_hierarchy_path(self):
        chunk = Chunk(
            text="Article content",
            metadata=ChunkMetadata(
                chunk_id="doc:chunk-000001",
                doc_id="doc",
                structure_type="article",
                structural_reference="Article 1",
                hierarchy_path=["Title I", "Chapter I", "Article 1"],
                source_pages=[1],
                source_unit_ids=["doc:page-000001"],
                segmentation_confidence=0.95,
                chunking_method="adaptive_rules_v1",
            ),
        )

        self.assertEqual(
            chunk.to_index_dict()["hierarchy_path"],
            ["Title I", "Chapter I", "Article 1"],
        )

    def test_index_defaults_to_candidate_and_guards_active_index(self):
        self.assertEqual(_resolve_index_output_dir(None, False), CANDIDATE_INDEX_DIR)
        self.assertEqual(_resolve_index_output_dir(None, True), INDEX_DIR)
        with self.assertRaises(ValueError):
            _resolve_index_output_dir(INDEX_DIR, False)
        with self.assertRaises(ValueError):
            _resolve_index_output_dir(Path("data/custom"), True)

    def test_inventory_refresh_preserves_completed_extraction_state(self):
        scanned = {"extraction_status": "pending"}
        existing = {
            "extraction_status": "done",
            "processing": {"status": "done"},
            "processed_at": "2026-01-01T00:00:00+00:00",
            "structured_path": "data/structured/doc.json",
            "unit_count": 12,
        }

        _preserve_processing_metadata(scanned, existing)

        self.assertEqual(scanned["extraction_status"], "done")
        self.assertEqual(scanned["unit_count"], 12)
        self.assertEqual(scanned["structured_path"], "data/structured/doc.json")


if __name__ == "__main__":
    unittest.main()
