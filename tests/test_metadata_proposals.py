"""Tests for source-linked document metadata proposals."""

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from normative_repository.config import PROJECT_ROOT
from normative_repository.metadata_proposals import (
    _collect_candidates,
    build_metadata_proposals,
)


class MetadataProposalTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT_ROOT / "data" / "review" / f".metadata-test-{uuid4().hex}"
        self.structured_dir = self.root / "structured"
        self.structured_dir.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)

    def test_table_of_contents_lines_are_not_suggested_as_titles(self):
        proposals = _collect_candidates([{
            "unit_id": "unit-000001",
            "unit_type": "document_section",
            "page_number": None,
            "text": "INDICE\nCAPÍTULO I\t8\nINTRODUCCIÓN\t7",
        }], max_units=1)

        self.assertEqual(proposals["title_candidates"], [])

    def test_proposals_keep_source_evidence_and_never_mutate_registry(self):
        doc_id = "doc-1"
        registry_path = self.root / "registry.json"
        registry_path.write_text(json.dumps({
            "records": {
                doc_id: {
                    "doc_id": doc_id,
                    "filename": "constitution.pdf",
                    "processing": {"status": "done"},
                    "reviewed_classification": "legislation",
                },
            },
        }), encoding="utf-8")
        original_registry = registry_path.read_bytes()
        (self.structured_dir / f"{doc_id}.json").write_text(json.dumps({
            "doc_id": doc_id,
            "units": [{
                "unit_id": "unit-000001",
                "unit_type": "page",
                "page_number": 1,
                "text": (
                    "CONSTITUCIÓN POLÍTICA DEL PERÚ\n"
                    "Promulgada el 29 de diciembre de 1993\n"
                    "Ley N° 123-2020\n"
                    "Congreso de la República"
                ),
            }],
        }, ensure_ascii=False), encoding="utf-8")
        output_path = self.root / "metadata-proposals.json"

        result = build_metadata_proposals(
            registry_path=registry_path,
            structured_dir=self.structured_dir,
            output_path=output_path,
        )

        self.assertFalse(result["registry_mutated"])
        self.assertEqual(registry_path.read_bytes(), original_registry)
        self.assertEqual(result["documents_examined"], 1)
        proposals = result["documents"][0]["proposals"]
        self.assertTrue(proposals["title_candidates"])
        self.assertEqual(
            proposals["document_identifier_candidates"][0]["value"],
            "Ley N° 123-2020",
        )
        self.assertEqual(
            proposals["date_mentions"][0]["source_unit_id"],
            "unit-000001",
        )
        self.assertEqual(
            proposals["issuing_authority_mentions"][0]["review_status"],
            "suggested_requires_review",
        )
        self.assertIn("legal_status", result["documents"][0]["fields_not_inferred"])

    def test_missing_structured_source_is_reported_not_fabricated(self):
        doc_id = "doc-missing"
        registry_path = self.root / "registry.json"
        registry_path.write_text(json.dumps({
            "records": {
                doc_id: {
                    "doc_id": doc_id,
                    "filename": "missing.pdf",
                    "processing": {"status": "done"},
                },
            },
        }), encoding="utf-8")

        result = build_metadata_proposals(
            registry_path=registry_path,
            structured_dir=self.structured_dir,
            output_path=self.root / "report.json",
        )

        self.assertEqual(result["documents_missing_structured_source"], 1)
        self.assertEqual(result["documents"][0]["status"], "structured_source_unavailable")
        self.assertNotIn("proposals", result["documents"][0])


if __name__ == "__main__":
    unittest.main()
