"""Tests for offline legal retrieval evaluation metrics."""

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from normative_repository.config import PROJECT_ROOT
from normative_repository.rag.evaluation import evaluate_retrieval_dataset


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT_ROOT / "data" / "review" / f".rag-eval-test-{uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.dataset = self.root / "queries.jsonl"

    def test_computes_ranked_metrics_from_explicit_labels(self):
        self.dataset.write_text(json.dumps({
            "query_id": "q1",
            "query": "article one",
            "relevant_sources": {"doc:chunk-1": 3, "doc:chunk-2": 1},
            "label_provenance": "review-1",
        }) + "\n", encoding="utf-8")

        report = evaluate_retrieval_dataset(
            self.dataset,
            retrieve=lambda query, limit: {
                "sources": [
                    {"doc_id": "doc", "chunk_id": "doc:chunk-2", "source_pages": [1]},
                    {"doc_id": "doc", "chunk_id": "doc:chunk-1", "source_unit_ids": ["unit-1"]},
                ][:limit],
            },
            k_values=(1, 2),
        )

        self.assertEqual(report["dataset_query_count"], 1)
        self.assertEqual(report["metrics"]["mean_reciprocal_rank"], 1.0)
        self.assertEqual(report["metrics"]["at_k"]["1"]["precision"], 1.0)
        self.assertEqual(report["metrics"]["at_k"]["1"]["recall"], 0.5)
        self.assertAlmostEqual(report["metrics"]["at_k"]["1"]["ndcg"], 1 / 7)
        self.assertEqual(report["metrics"]["citation_traceability"], 1.0)
        self.assertFalse(report["legal_correctness_claimed"])

    def test_rejects_missing_gold_and_empty_datasets(self):
        self.dataset.write_text(json.dumps({
            "query_id": "q1",
            "query": "article one",
            "relevant_sources": {},
        }) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            evaluate_retrieval_dataset(self.dataset, retrieve=lambda query, limit: {})

        self.dataset.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            evaluate_retrieval_dataset(self.dataset, retrieve=lambda query, limit: {})


if __name__ == "__main__":
    unittest.main()
