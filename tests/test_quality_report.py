"""Tests for operational quality-report helpers."""

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4
from pypdf import PdfWriter

from normative_repository.config import PROJECT_ROOT
from normative_repository.index import _build_document_metadata
from normative_repository.quality_report import (
    _candidate_index_findings,
    _compare_pdf_text_layer,
    _scan_source_files,
)


class QualityReportTests(unittest.TestCase):
    def make_workspace_temp_dir(self):
        path = PROJECT_ROOT / "data" / "review" / f".quality-test-{uuid4().hex}"
        path.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, path)
        return path

    def test_source_scan_counts_unsupported_extensions(self):
        root = self.make_workspace_temp_dir()
        (root / "source.pdf").write_bytes(b"pdf")
        (root / "notes.txt").write_text("notes", encoding="utf-8")

        report = _scan_source_files(root)

        self.assertEqual(report["files_seen"], 2)
        self.assertEqual(report["supported_files_seen"], 1)
        self.assertEqual(report["unsupported_count"], 1)
        self.assertEqual(report["unsupported_examples"], ["notes.txt"])

    def test_pdf_text_layer_comparison_counts_replacements_by_markdown_page(self):
        root = self.make_workspace_temp_dir()
        source = root / "source.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with source.open("wb") as stream:
            writer.write(stream)

        comparison = _compare_pdf_text_layer(
            source,
            [{"page_number": 1, "text": "Recovered � text"}],
            "## Page 1\n\nRecovered � text\n",
        )

        self.assertEqual(comparison["status"], "differs_from_pdf_text_layer")
        self.assertEqual(comparison["affected_pages"][0]["markdown_count"], 1)

    def test_candidate_index_reference_audit(self):
        root = self.make_workspace_temp_dir()
        document_metadata = _build_document_metadata(
            {
                "doc_id": "doc",
                "filename": "source.pdf",
                "relative_path": "source.pdf",
                "sha256": "d" * 64,
                "extension": ".pdf",
                "size_bytes": 100,
            },
            reviewed_classification=None,
            classification_status="pending_review",
        )
        (root / "documents.jsonl").write_text(
            json.dumps({
                "doc_id": "doc",
                "filename": "source.pdf",
                "document_metadata": document_metadata,
            }) + "\n",
            encoding="utf-8",
        )
        (root / "units.jsonl").write_text(
            json.dumps({"doc_id": "doc", "unit_id": "doc:page-000001", "page_number": 1}) + "\n",
            encoding="utf-8",
        )
        (root / "chunks.jsonl").write_text(
            json.dumps({
                "chunk_id": "doc:chunk-000001",
                "doc_id": "doc",
                "structure_type": "article",
                "structure_reference": "Article 1",
                "hierarchy_path": ["Article 1"],
                "source_pages": [1],
                "source_unit_ids": ["doc:page-000001"],
                "text": "Source passage",
                "segmentation_confidence": 0.95,
                "chunking_method": "adaptive_rules_v1",
            }) + "\n",
            encoding="utf-8",
        )
        (root / "index-report.json").write_text(
            json.dumps({
                "documents_indexed": 1,
                "units_written": 1,
                "chunks_written": 1,
                "documents_skipped": 0,
                "source_text_coverage": {"coverage_ratio": 1.0},
            }),
            encoding="utf-8",
        )

        report = _candidate_index_findings(root)

        self.assertTrue(report["available"])
        self.assertEqual(report["chunks_read"], 1)
        self.assertEqual(report["schema_invalid_chunks"], 0)
        self.assertEqual(report["missing_document_references"], 0)
        self.assertEqual(report["missing_unit_references"], 0)
        self.assertEqual(report["invalid_page_references"], 0)
        self.assertEqual(report["hierarchy_path_mismatches"], 0)
        self.assertEqual(report["documents_missing_metadata_schema"], 0)
        self.assertEqual(report["invalid_document_metadata"], 0)


if __name__ == "__main__":
    unittest.main()
