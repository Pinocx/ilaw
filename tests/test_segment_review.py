"""Tests for exhaustive page-based segmentation review and evaluation."""

import csv
import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from normative_repository.config import PROJECT_ROOT
from normative_repository.segment_review import (
    GOLD_SEGMENT_FIELDS,
    PAGE_REVIEW_FIELDS,
    build_segment_review_packet,
    evaluate_segment_review_packet,
)


class SegmentReviewTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT_ROOT / "data" / "review" / f".segment-review-test-{uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.structured_dir = self.root / "structured"
        self.structured_dir.mkdir()
        self.packet_dir = self.root / "packet"
        self.index_dir = self.root / "index"
        self.index_dir.mkdir()
        self.doc_id = "doc-1"
        self.filename = "Sample.pdf"
        self.page_text = "CAPÍTULO I\nArtículo 1.- Regla\nTexto base."
        self.unit = {
            "unit_id": "doc-1:page-000001",
            "page_number": 1,
            "text": self.page_text,
        }
        (self.structured_dir / "doc-1.json").write_text(
            json.dumps({"units": [self.unit]}), encoding="utf-8"
        )
        self.review_csv = self.root / "review-labels.csv"
        with self.review_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["document_id", "filename", "source_format"],
            )
            writer.writeheader()
            writer.writerow({
                "document_id": self.doc_id,
                "filename": self.filename,
                "source_format": ".pdf",
            })

    def write_packet_inputs(self, *, complete="yes", fidelity="yes", gold=None):
        task_id = f"{self.doc_id}:page-000001"
        task = {
            "task_id": task_id,
            "document_id": self.doc_id,
            "filename": self.filename,
            "page_number": 1,
            "source_unit_id": self.unit["unit_id"],
            "page_text_sha256": __import__("hashlib").sha256(
                self.page_text.encode("utf-8")
            ).hexdigest(),
            "token_count": len(self.page_text.split()),
        }
        self.packet_dir.mkdir(exist_ok=True)
        (self.packet_dir / "manifest.json").write_text(
            json.dumps({"tasks": [task]}), encoding="utf-8"
        )
        with (self.packet_dir / "page-review.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=PAGE_REVIEW_FIELDS)
            writer.writeheader()
            writer.writerow({
                **{key: task[key] for key in PAGE_REVIEW_FIELDS[:6]},
                "page_text_fidelity": fidelity,
                "exhaustive_annotation_complete": complete,
                "review_notes": "",
            })
        with (self.packet_dir / "gold-segments.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=GOLD_SEGMENT_FIELDS)
            writer.writeheader()
            writer.writerows(gold or [])
        chunks = [
            {"doc_id": self.doc_id, "chunk_id": "doc-1:chunk-1", "structure_type": "chapter", "structure_reference": "CAPÍTULO I", "parent_reference": None, "hierarchy_path": ["CAPÍTULO I"], "part_number": 1, "part_count": 1, "text": "CAPÍTULO I"},
            {"doc_id": self.doc_id, "chunk_id": "doc-1:chunk-2", "structure_type": "article", "structure_reference": "Artículo 1", "parent_reference": "CAPÍTULO I", "hierarchy_path": ["CAPÍTULO I", "Artículo 1"], "part_number": 1, "part_count": 1, "text": "Artículo 1.- Regla"},
            {"doc_id": self.doc_id, "chunk_id": "doc-1:chunk-3", "structure_type": "paragraph", "structure_reference": None, "parent_reference": "Artículo 1", "hierarchy_path": ["CAPÍTULO I", "Artículo 1"], "part_number": 1, "part_count": 1, "text": "Texto base."},
        ]
        (self.index_dir / "chunks.jsonl").write_text(
            "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
            encoding="utf-8",
        )
        return task_id

    def gold_row(self, task_id, order, text, structure_type):
        return {
            "task_id": task_id,
            "segment_order": order,
            "segment_text": text,
            "structure_type": structure_type,
            "structural_reference": "" if structure_type == "paragraph" else text.split()[0],
            "parent_reference": "",
            "hierarchy_path": "[]",
            "review_notes": "",
        }

    def test_exhaustive_page_annotation_measures_omission_and_exact_spans(self):
        task_id = f"{self.doc_id}:page-000001"
        gold = [
            self.gold_row(task_id, 1, "CAPÍTULO I", "chapter"),
            self.gold_row(task_id, 2, "Artículo 1.- Regla", "article"),
            self.gold_row(task_id, 3, "Texto base.", "paragraph"),
        ]
        self.write_packet_inputs(gold=gold)

        report = evaluate_segment_review_packet(
            packet_dir=self.packet_dir,
            structured_dir=self.structured_dir,
            index_dir=self.index_dir,
            output_path=self.root / "evaluation.json",
        )

        self.assertEqual(report["segment_level"]["status"], "measured")
        self.assertEqual(report["segment_level"]["precision"], 1.0)
        self.assertEqual(report["segment_level"]["recall"], 1.0)
        self.assertEqual(report["segment_level"]["f1"], 1.0)
        self.assertEqual(report["token_level_structure_labels"]["accuracy"], 1.0)

    def test_packet_samples_pages_deterministically_and_never_overwrites_reviews(self):
        result = build_segment_review_packet(
            review_csv=self.review_csv,
            structured_dir=self.structured_dir,
            output_dir=self.packet_dir,
            pages_per_document=1,
            seed=10,
        )
        self.assertEqual(result["task_count"], 1)
        with (self.packet_dir / "page-review.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["page_text_fidelity"] = "yes"
        with (self.packet_dir / "page-review.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=PAGE_REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaises(FileExistsError):
            build_segment_review_packet(
                review_csv=self.review_csv,
                structured_dir=self.structured_dir,
                output_dir=self.packet_dir,
                pages_per_document=1,
                seed=10,
            )

    def test_incomplete_annotation_does_not_claim_metrics_or_read_index(self):
        self.write_packet_inputs(complete="no", fidelity="")

        report = evaluate_segment_review_packet(
            packet_dir=self.packet_dir,
            structured_dir=self.structured_dir,
            index_dir=self.root / "missing-index",
            output_path=self.root / "evaluation.json",
        )

        self.assertEqual(report["segment_level"]["status"], "awaiting_complete_human_annotations")
        self.assertIsNone(report["segment_level"]["f1"])
        self.assertEqual(report["evaluated_page_count"], 0)


if __name__ == "__main__":
    unittest.main()
