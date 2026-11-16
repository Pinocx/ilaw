"""Tests for the official document metadata promotion policy."""

import unittest

from pydantic import ValidationError

from normative_repository.index import _build_document_metadata
from normative_repository.schemas import DocumentMetadata


class DocumentMetadataPolicyTests(unittest.TestCase):
    def test_registry_identity_and_human_reviewed_type_are_official(self):
        record = {
            "doc_id": "doc-1",
            "filename": "source.pdf",
            "relative_path": "laws/source.pdf",
            "sha256": "a" * 64,
            "extension": ".pdf",
            "size_bytes": 1234,
            "reviewed_classification": "legislation",
            "classification_reviewed_at": "2026-11-14T12:00:00+00:00",
            "classification_review_source": "review-labels.csv",
            "classification_review_decision": "corrected_suggestion",
        }

        metadata = _build_document_metadata(
            record,
            reviewed_classification="legislation",
            classification_status="reviewed",
        )

        self.assertEqual(metadata["source_sha256"], "a" * 64)
        self.assertEqual(metadata["document_type"], "legislation")
        self.assertEqual(metadata["field_provenance"]["source_sha256"]["status"], "system_verified")
        self.assertEqual(metadata["field_provenance"]["document_type"]["status"], "human_verified")
        self.assertIsNone(metadata["document_title"])
        self.assertIsNone(metadata["jurisdiction"])
        self.assertEqual(metadata["legal_status"], "unknown")
        self.assertEqual(metadata["field_provenance"]["document_type"]["review_decision"], "corrected_suggestion")
        DocumentMetadata.model_validate(metadata)

    def test_suggested_type_is_not_promoted_to_official_metadata(self):
        metadata = _build_document_metadata(
            {
                "doc_id": "doc-2",
                "filename": "source.pdf",
                "relative_path": "source.pdf",
                "sha256": "b" * 64,
                "extension": ".pdf",
                "size_bytes": 456,
                "suggested_classification": "legislation",
            },
            reviewed_classification=None,
            classification_status="pending_review",
        )

        self.assertIsNone(metadata["document_type"])
        self.assertEqual(metadata["classification_status"], "pending_review")
        self.assertEqual(metadata["field_provenance"]["document_type"]["status"], "pending_review")

    def test_semantic_metadata_cannot_be_official_without_human_evidence(self):
        metadata = _build_document_metadata(
            {
                "doc_id": "doc-3",
                "filename": "source.pdf",
                "relative_path": "source.pdf",
                "sha256": "c" * 64,
                "extension": ".pdf",
                "size_bytes": 789,
            },
            reviewed_classification=None,
            classification_status="pending_review",
        )
        metadata["publication_date"] = "1993-12-29"

        with self.assertRaises(ValidationError):
            DocumentMetadata.model_validate(metadata)


if __name__ == "__main__":
    unittest.main()
