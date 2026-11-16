"""Tests for offline human-reviewed model comparisons."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from normative_repository.llm import evaluate_model_benchmarks


class ModelEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).parent / "fixtures" / "model-benchmark-test.jsonl"
        self.addCleanup(self.path.unlink, missing_ok=True)

    def test_reports_quality_citation_compliance_latency_and_usage_metrics(self) -> None:
        rows = [
            {"record_type": "metadata", "schema_version": "1", "label_provenance": "review batch A"},
            {
                "query_id": "q1", "run_id": "r1", "provider": "mock", "model": "m1",
                "expected_source_ids": ["d::c"], "generated_source_ids": ["d::c", "x::y"],
                "citations_verified": True, "instruction_compliance": True,
                "legal_quality_rating": 4, "latency_ms": 1000, "input_tokens": 10,
                "output_tokens": 5, "estimated_cost": 0.01, "human_reviewed": True,
            },
        ]
        self.path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        report = evaluate_model_benchmarks(self.path)
        result = report["models"][0]
        self.assertEqual(result["citation_precision"], 0.5)
        self.assertEqual(result["citation_recall"], 1.0)
        self.assertEqual(result["mean_legal_quality_rating"], 4)
        self.assertEqual(result["mean_input_tokens"], 10)
        self.assertFalse(report["legal_correctness_claim"])

    def test_rejects_unreviewed_or_unlabeled_records(self) -> None:
        rows = [
            {"record_type": "metadata", "schema_version": "1", "label_provenance": "reviewer"},
            {"query_id": "q", "run_id": "r", "provider": "mock", "model": "m", "citations_verified": True, "instruction_compliance": True, "human_reviewed": False},
        ]
        self.path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        with self.assertRaises(ValueError):
            evaluate_model_benchmarks(self.path)


if __name__ == "__main__":
    unittest.main()
