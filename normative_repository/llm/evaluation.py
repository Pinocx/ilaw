"""Offline model benchmark metrics from explicitly reviewed response records."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelBenchmarkRecord(BaseModel):
    """One reviewed model response and its evaluator labels."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=256)
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=256)
    expected_source_ids: list[str] = Field(default_factory=list, max_length=100)
    generated_source_ids: list[str] = Field(default_factory=list, max_length=100)
    citations_verified: bool
    instruction_compliance: bool
    legal_quality_rating: int | None = Field(default=None, ge=1, le=5)
    latency_ms: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    human_reviewed: bool


def evaluate_model_benchmarks(dataset_path: Path) -> dict[str, Any]:
    """Aggregate model quality, citation, compliance, latency, token, and cost metrics."""
    records: list[ModelBenchmarkRecord] = []
    metadata = None
    try:
        with dataset_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}.") from exc
                if metadata is None:
                    if item.get("record_type") != "metadata" or item.get("schema_version") != "1":
                        raise ValueError("The first dataset record must declare schema_version 1 metadata.")
                    provenance = item.get("label_provenance")
                    if not isinstance(provenance, str) or not provenance.strip():
                        raise ValueError("Dataset label_provenance must identify the human evaluation source.")
                    metadata = {"schema_version": "1", "label_provenance": provenance}
                    continue
                record = ModelBenchmarkRecord.model_validate(item)
                if not record.human_reviewed:
                    raise ValueError(f"Benchmark record on line {line_number} is not human reviewed.")
                records.append(record)
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read model benchmark dataset: {dataset_path}") from exc
    if not records:
        raise ValueError("Model benchmark dataset contains no reviewed response records.")

    groups: dict[tuple[str, str], list[ModelBenchmarkRecord]] = defaultdict(list)
    for record in records:
        groups[(record.provider, record.model)].append(record)
    results = []
    for (provider, model), samples in sorted(groups.items()):
        citation_precisions = []
        citation_recalls = []
        citation_f1s = []
        for sample in samples:
            expected = set(sample.expected_source_ids)
            generated = set(sample.generated_source_ids)
            common = len(expected & generated)
            precision = common / len(generated) if generated else 0.0
            recall = common / len(expected) if expected else 0.0
            citation_precisions.append(precision)
            citation_recalls.append(recall)
            citation_f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        quality_scores = [sample.legal_quality_rating for sample in samples if sample.legal_quality_rating is not None]
        input_tokens = [sample.input_tokens for sample in samples if sample.input_tokens is not None]
        output_tokens = [sample.output_tokens for sample in samples if sample.output_tokens is not None]
        costs = [sample.estimated_cost for sample in samples if sample.estimated_cost is not None]
        results.append({
            "provider": provider,
            "model": model,
            "sample_count": len(samples),
            "citation_precision": sum(citation_precisions) / len(samples),
            "citation_recall": sum(citation_recalls) / len(samples),
            "citation_f1": sum(citation_f1s) / len(samples),
            "citation_verification_rate": sum(sample.citations_verified for sample in samples) / len(samples),
            "instruction_compliance_rate": sum(sample.instruction_compliance for sample in samples) / len(samples),
            "mean_legal_quality_rating": sum(quality_scores) / len(quality_scores) if quality_scores else None,
            "legal_quality_review_count": len(quality_scores),
            "mean_latency_ms": sum(sample.latency_ms for sample in samples) / len(samples),
            "mean_input_tokens": sum(input_tokens) / len(input_tokens) if input_tokens else None,
            "mean_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else None,
            "total_estimated_cost": sum(costs) if costs else None,
        })
    return {
        "schema_version": "1",
        "label_provenance": metadata["label_provenance"],
        "reviewed_sample_count": len(records),
        "model_count": len(results),
        "models": results,
        "legal_correctness_claim": False,
    }
