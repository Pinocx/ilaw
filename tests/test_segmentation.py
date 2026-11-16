"""Regression tests for deterministic legal segmentation and source coverage."""

import unittest

from normative_repository.index import _coverage_metrics
from normative_repository.review_packet import _associated_unit_text
from normative_repository.schemas import AISegment, DocumentType, SegmentType, ValidationStatus
from normative_repository.segmentation import _detect_marker, segment_document
from normative_repository.validators import DeterministicValidator


class MarkerDetectionTests(unittest.TestCase):
    def test_legislative_article_is_detected_only_for_legislation(self):
        line = "Artículo 1.- Objeto de la ley"

        self.assertEqual(_detect_marker(line, "legislation")["kind"], "article")
        self.assertIsNone(_detect_marker(line, "legal_doctrine"))

    def test_standalone_provision_in_doctrine_is_quoted(self):
        marker = _detect_marker("Artículo 2", "legal_doctrine")

        self.assertEqual(marker["kind"], "quoted_provision")
        self.assertEqual(marker["reference"], "Artículo 2")

    def test_table_of_contents_entry_is_not_an_article(self):
        marker = _detect_marker(
            "Art. 1 Principio de legalidad ........ 38",
            "legislation",
        )

        self.assertEqual(marker["kind"], "table_of_contents_entry")

    def test_tab_separated_page_number_marks_a_table_of_contents_entry(self):
        marker = _detect_marker("CAPÍTULO I\t8", "legal_doctrine")

        self.assertEqual(marker["kind"], "table_of_contents_entry")
        self.assertEqual(marker["reference"], "CAPÍTULO I")

    def test_article_citation_is_not_a_heading(self):
        self.assertIsNone(
            _detect_marker("Artículo 1 del Código Penal", "legislation")
        )

    def test_jurisprudence_criterion_marker_is_recognized(self):
        marker = _detect_marker("J 073", "jurisprudence")

        self.assertEqual(marker["kind"], "case_section")
        self.assertEqual(marker["reference"], "Criterio J 073")


class SegmentationTests(unittest.TestCase):
    def test_article_chunk_carries_full_hierarchy_path(self):
        chunks = segment_document(
            doc_id="doc",
            classification="legislation",
            units=[{
                "unit_id": "doc:page-000001",
                "page_number": 1,
                "text": (
                    "TÍTULO I\nDISPOSICIONES GENERALES\n"
                    "CAPÍTULO I\nReglas básicas\n"
                    "Artículo 1.- Objeto\nLa presente ley establece reglas."
                ),
            }],
        )
        article = next(chunk for chunk in chunks if chunk["structure_type"] == "article")

        self.assertEqual(
            article["hierarchy_path"],
            ["TÍTULO I", "CAPÍTULO I", "Artículo 1"],
        )
        self.assertEqual(article["source_unit_ids"], ["doc:page-000001"])
        self.assertEqual(article["source_pages"], [1])

    def test_doctrine_does_not_emit_actual_article_chunks(self):
        chunks = segment_document(
            doc_id="doctrine",
            classification="legal_doctrine",
            units=[{
                "unit_id": "doctrine:page-000001",
                "page_number": 1,
                "text": "Artículo 1\nTexto legal citado.\nCOMENTARIO:\nAnálisis del autor.",
            }],
        )

        self.assertNotIn("article", {chunk["structure_type"] for chunk in chunks})
        self.assertIn("quoted_provision", {chunk["structure_type"] for chunk in chunks})
        self.assertIn("commentary", {chunk["structure_type"] for chunk in chunks})


class SourceCoverageTests(unittest.TestCase):
    def test_coverage_is_whitespace_insensitive_and_ordered(self):
        exact = _coverage_metrics(
            "Artículo 1.\nLa ley dispone esto.",
            ["Artículo 1. La ley", "dispone esto."],
        )
        missing = _coverage_metrics(
            "alpha beta gamma delta",
            ["alpha beta delta"],
        )

        self.assertTrue(exact["exact_reconstruction"])
        self.assertEqual(exact["coverage_ratio"], 1.0)
        self.assertFalse(missing["exact_reconstruction"])
        self.assertLess(missing["coverage_ratio"], 1.0)

    def test_validator_checks_text_beyond_legacy_length_limit(self):
        source_text = ("background " * 5000) + "verified source passage"
        segment = AISegment(
            segment_id="segment-1",
            document_id="doc",
            segment_type=SegmentType.OTHER,
            source_pages=[1],
            source_unit_ids=["doc:page-000001"],
            text="verified source passage",
            classification=DocumentType.UNKNOWN,
            confidence=0.95,
            prompt_version="test",
            model_name="test",
        )

        validated, report = DeterministicValidator().validate_segments(
            [segment], "doc", source_text
        )

        self.assertEqual(report.failed_segments, 0)
        self.assertEqual(validated[0].validation_status, ValidationStatus.VALIDATED)
        self.assertEqual(segment.validation_status, ValidationStatus.PENDING_REVIEW)


class ReviewPacketTests(unittest.TestCase):
    def test_packet_resolves_a_specific_page_unit(self):
        sample = {
            "record": {"doc_id": "doc"},
            "payload": {"units": [
                {"unit_id": "unit-000001", "page_number": 1, "text": "first page"},
                {"unit_id": "unit-000002", "page_number": 2, "text": "second page passage"},
            ]},
            "chunk": {
                "source_pages": [2],
                "source_unit_ids": ["doc:page-000002"],
                "text": "second page passage",
            },
        }

        units, trace_match = _associated_unit_text(sample)

        self.assertEqual([unit["page_number"] for unit in units], [2])
        self.assertTrue(trace_match)


if __name__ == "__main__":
    unittest.main()
