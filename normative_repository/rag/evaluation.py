"""Offline retrieval metrics for reviewer-labeled legal query datasets."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _dcg(relevance: list[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(relevance, start=1)
    )


def evaluate_retrieval_dataset(
    dataset_path: Path,
    *,
    retrieve: Callable[[str, int], dict[str, Any]],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Compute retrieval metrics from explicit graded source labels."""
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers.")
    dataset_path = Path(dataset_path)
    cases: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(case, dict):
                raise ValueError(f"Dataset line {line_number} must contain a JSON object.")
            query_id = case.get("query_id")
            query = case.get("query")
            relevant = case.get("relevant_sources")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"Dataset line {line_number} requires query_id.")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"Dataset line {line_number} requires query.")
            if not isinstance(relevant, dict) or not relevant:
                raise ValueError(
                    f"Dataset line {line_number} requires non-empty relevant_sources "
                    "mapping chunk IDs to positive integer relevance grades."
                )
            for source_id, grade in relevant.items():
                if not isinstance(source_id, str) or not source_id.strip():
                    raise ValueError(f"Dataset line {line_number} contains an invalid source ID.")
                if not isinstance(grade, int) or isinstance(grade, bool) or grade < 1:
                    raise ValueError(f"Dataset line {line_number} relevance grades must be positive integers.")
            cases.append(case)

    if not cases:
        raise ValueError("Retrieval evaluation dataset is empty; no metrics can be computed.")
    query_ids = [case["query_id"] for case in cases]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Retrieval evaluation query_id values must be unique.")

    per_query: list[dict[str, Any]] = []
    for case in cases:
        max_k = max(k_values)
        result = retrieve(case["query"], max_k)
        source_rows = result.get("sources") or result.get("results") or []
        retrieved_ids = [
            str(row.get("chunk_id"))
            for row in source_rows
            if row.get("chunk_id") is not None
        ]
        relevant_sources: dict[str, int] = case["relevant_sources"]
        relevant_ids = set(relevant_sources)
        reciprocal_rank = next(
            (1 / rank for rank, source_id in enumerate(retrieved_ids, start=1) if source_id in relevant_ids),
            0.0,
        )
        metrics_at_k: dict[str, dict[str, float]] = {}
        for k in k_values:
            top_ids = retrieved_ids[:k]
            hits = sum(source_id in relevant_ids for source_id in top_ids)
            recall = hits / len(relevant_ids)
            precision = hits / k
            observed_grades = [relevant_sources.get(source_id, 0) for source_id in top_ids]
            ideal_grades = sorted(relevant_sources.values(), reverse=True)[:k]
            ideal_grades.extend([0] * (k - len(ideal_grades)))
            ideal_dcg = _dcg(ideal_grades)
            ndcg = _dcg(observed_grades) / ideal_dcg if ideal_dcg else 0.0
            metrics_at_k[str(k)] = {
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
            }

        traceable = sum(
            bool(row.get("doc_id") and row.get("chunk_id") and (
                row.get("source_unit_ids") or row.get("source_pages")
            ))
            for row in source_rows
        )
        per_query.append({
            "query_id": case["query_id"],
            "retrieved_count": len(retrieved_ids),
            "relevant_source_count": len(relevant_ids),
            "reciprocal_rank": reciprocal_rank,
            "metrics_at_k": metrics_at_k,
            "retrieval_coverage": 1.0 if retrieved_ids else 0.0,
            "citation_traceability": traceable / len(source_rows) if source_rows else 0.0,
        })

    denominator = len(per_query)
    aggregate: dict[str, Any] = {
        "query_count": denominator,
        "mean_reciprocal_rank": sum(row["reciprocal_rank"] for row in per_query) / denominator,
        "retrieval_coverage": sum(row["retrieval_coverage"] for row in per_query) / denominator,
        "citation_traceability": sum(row["citation_traceability"] for row in per_query) / denominator,
        "at_k": {},
    }
    for k in k_values:
        aggregate["at_k"][str(k)] = {
            metric: sum(row["metrics_at_k"][str(k)][metric] for row in per_query) / denominator
            for metric in ("precision", "recall", "ndcg")
        }

    return {
        "schema_version": "1.0",
        "evaluation_scope": "Offline retrieval metrics against explicitly labeled relevant chunk IDs.",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_query_count": len(cases),
        "k_values": list(k_values),
        "label_provenance": "Read from the supplied evaluation dataset; this evaluator does not create or modify labels.",
        "metrics": aggregate,
        "queries": per_query,
        "legal_correctness_claimed": False,
    }
