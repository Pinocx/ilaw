"""Tests for reviewer-label validation and evaluation reporting."""

import csv
import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from normative_repository.config import PROJECT_ROOT
from normative_repository.evaluation import (
    apply_reviewed_classifications,
    evaluate_review_labels,
)


FIELDS = [
    "document_id",
    "filename",
    "structured_suggested_classification",
    "existing_human_classification",
    "classification_accepted",
    "corrected_classification",
    "source_text_fidelity",
    "structure_label_correct",
    "hierarchy_correct",
    "page_reference_correct",
    "review_notes",
]


class ReviewEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = (
            PROJECT_ROOT / "data" / "review" / f".evaluation-test-{uuid4().hex}"
        )
        self.temp_dir.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.temp_dir)
        self.input_path = self.temp_dir / "labels.csv"
        self.output_path = self.temp_dir / "report.json"

    def write_rows(self, rows):
        with self.input_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, **overrides):
        values = {field: "" for field in FIELDS}
        values.update({
            "document_id": "doc-1",
            "filename": "Sample.pdf",
            "structured_suggested_classification": "legislation",
        })
        values.update(overrides)
        return values

    def test_reports_supported_metrics_without_claiming_segment_f1(self):
        self.write_rows([
            self.row(
                existing_human_classification="jurisprudence",
                classification_accepted="no",
                corrected_classification="jurisprudence",
                source_text_fidelity="yes",
                structure_label_correct="partial",
                hierarchy_correct="no",
                page_reference_correct="not_applicable",
            )
        ])

        report = evaluate_review_labels(self.input_path, self.output_path)
        persisted = json.loads(self.output_path.read_text(encoding="utf-8"))

        self.assertEqual(report["new_classification_decision_counts"], {"no": 1})
        self.assertEqual(report["review_field_summary"]["source_text_fidelity"]["affirmative_share"], 1.0)
        self.assertEqual(report["segmentation_evaluation"]["status"], "insufficient_segment_ground_truth")
        self.assertIsNone(report["segmentation_evaluation"]["f1"])
        self.assertFalse(report["ground_truth_mutated"])
        self.assertFalse(report["registry_mutated"])
        self.assertEqual(persisted["packet_document_count"], 1)

    def test_unreviewed_packet_reports_no_classification_accuracy(self):
        self.write_rows([self.row()])

        report = evaluate_review_labels(self.input_path, self.output_path)

        self.assertEqual(
            report["existing_label_classification_baseline"]["documents_with_existing_human_labels"],
            0,
        )
        self.assertIsNone(report["existing_label_classification_baseline"]["accuracy"])
        self.assertEqual(report["documents_with_new_human_decisions"], 0)

    def test_rejects_unconfirmed_correction_without_writing_report(self):
        self.write_rows([self.row(classification_accepted="no")])

        with self.assertRaisesRegex(ValueError, "corrected_classification is required"):
            evaluate_review_labels(self.input_path, self.output_path)
        self.assertFalse(self.output_path.exists())

    def test_import_persists_only_explicit_human_classifications_idempotently(self):
        corrected = self.row(
            document_id="doc-2",
            filename="Second.pdf",
            structured_suggested_classification="legislation",
            classification_accepted="no",
            corrected_classification="legal_doctrine",
        )
        blank = self.row(
            document_id="doc-3",
            filename="Third.pdf",
            classification_accepted="",
        )
        self.write_rows([
            self.row(classification_accepted="yes"),
            corrected,
            blank,
        ])
        registry_path = self.temp_dir / "registry.json"
        registry_path.write_text(json.dumps({
            "schema_version": 1,
            "records": {
                "doc-1": {
                    "filename": "Sample.pdf",
                    "reviewed_classification": None,
                    "classification_review_required": True,
                    "review": {"classification": "pending"},
                },
                "doc-2": {
                    "filename": "Second.pdf",
                    "reviewed_classification": None,
                    "classification_review_required": True,
                    "review": {"classification": "pending"},
                },
                "doc-3": {
                    "filename": "Third.pdf",
                    "reviewed_classification": None,
                    "classification_review_required": True,
                },
            },
        }), encoding="utf-8")

        first = apply_reviewed_classifications(
            self.input_path,
            registry_path,
            self.output_path,
        )
        second = apply_reviewed_classifications(
            self.input_path,
            registry_path,
            self.output_path,
        )
        persisted = json.loads(registry_path.read_text(encoding="utf-8"))

        self.assertEqual(first["classifications_applied"], 2)
        self.assertEqual(first["already_matching_human_labels"], 0)
        self.assertEqual(second["classifications_applied"], 0)
        self.assertEqual(second["already_matching_human_labels"], 2)
        self.assertEqual(persisted["records"]["doc-1"]["reviewed_classification"], "legislation")
        self.assertEqual(persisted["records"]["doc-2"]["reviewed_classification"], "legal_doctrine")
        self.assertTrue(persisted["records"]["doc-3"]["classification_review_required"])

    def test_import_refuses_to_overwrite_an_existing_human_label(self):
        self.write_rows([self.row(
            existing_human_classification="jurisprudence",
            classification_accepted="no",
            corrected_classification="legal_doctrine",
        )])
        registry_path = self.temp_dir / "registry.json"
        registry_path.write_text(json.dumps({
            "schema_version": 1,
            "records": {
                "doc-1": {
                    "filename": "Sample.pdf",
                    "reviewed_classification": "jurisprudence",
                }
            },
        }), encoding="utf-8")
        before = registry_path.read_text(encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Refusing to overwrite existing human label"):
            apply_reviewed_classifications(self.input_path, registry_path, self.output_path)
        self.assertEqual(registry_path.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
