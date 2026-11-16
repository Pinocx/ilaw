"""Validate human review labels and report only metrics supported by evidence."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CATEGORIES, REGISTRY_FILE, REPORTS_DIR, RETROSPECTIVE_PILOT_DIR
from .registry import load_registry, save_registry


REQUIRED_FIELDS = {
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
}
CLASSIFICATION_DECISIONS = {"yes", "no"}
REVIEW_DECISIONS = {"yes", "no", "partial", "not_applicable"}
REVIEW_FIELDS = (
    "source_text_fidelity",
    "structure_label_correct",
    "hierarchy_correct",
    "page_reference_correct",
)


def _normalized(value: Any) -> str:
    return str(value or "").strip().casefold()


def _read_review_rows(input_path: Path) -> list[dict[str, str]]:
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError("Review CSV has no header row.")
            missing = REQUIRED_FIELDS - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    "Review CSV is missing required columns: "
                    + ", ".join(sorted(missing))
                )
            rows = [
                {key: str(value or "").strip() for key, value in row.items() if key}
                for row in reader
            ]
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read review CSV: {input_path}") from exc

    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        doc_id = row.get("document_id", "")
        if not doc_id:
            raise ValueError(f"Review CSV row {line_number} has no document_id.")
        if doc_id in seen:
            raise ValueError(f"Duplicate document_id in review CSV: {doc_id}")
        seen.add(doc_id)

        accepted = _normalized(row.get("classification_accepted"))
        if accepted and accepted not in CLASSIFICATION_DECISIONS:
            raise ValueError(
                f"Row {line_number}: classification_accepted must be yes or no."
            )
        corrected = _normalized(row.get("corrected_classification"))
        existing = _normalized(row.get("existing_human_classification"))
        suggestion = _normalized(row.get("structured_suggested_classification"))
        if accepted == "no" and corrected not in CATEGORIES:
            raise ValueError(
                f"Row {line_number}: a valid corrected_classification is required "
                "when classification_accepted is no."
            )
        if accepted == "yes" and corrected and corrected != suggestion:
            raise ValueError(
                f"Row {line_number}: corrected_classification conflicts with an "
                "accepted suggestion."
            )
        if existing and existing not in CATEGORIES:
            raise ValueError(
                f"Row {line_number}: invalid existing_human_classification."
            )
        if suggestion and suggestion not in CATEGORIES:
            raise ValueError(
                f"Row {line_number}: invalid structured_suggested_classification."
            )
        for field in REVIEW_FIELDS:
            decision = _normalized(row.get(field))
            if decision and decision not in REVIEW_DECISIONS:
                raise ValueError(
                    f"Row {line_number}: {field} must be one of "
                    + ", ".join(sorted(REVIEW_DECISIONS))
                    + "."
                )
    return rows


def _field_review_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field in REVIEW_FIELDS:
        counts = Counter(_normalized(row.get(field)) for row in rows)
        counts.pop("", None)
        applicable = sum(
            count for decision, count in counts.items()
            if decision != "not_applicable"
        )
        summary[field] = {
            "decision_counts": dict(sorted(counts.items())),
            "reviewed_documents": sum(counts.values()),
            "applicable_documents": applicable,
            "affirmative_share": (
                counts.get("yes", 0) / applicable if applicable else None
            ),
        }
    return summary


def _classification_baseline(rows: list[dict[str, str]]) -> dict[str, Any]:
    assessed: list[dict[str, str]] = []
    for row in rows:
        suggestion = _normalized(row.get("structured_suggested_classification"))
        reference = _normalized(row.get("existing_human_classification"))
        if suggestion and reference:
            assessed.append({
                "document_id": row["document_id"],
                "filename": row.get("filename", ""),
                "suggestion": suggestion,
                "human_reference": reference,
                "matches": suggestion == reference,
            })
    correct = sum(item["matches"] for item in assessed)
    return {
        "sample_scope": "selected retrospective pilot; not representative",
        "documents_with_existing_human_labels": len(assessed),
        "matches": correct,
        "accuracy": correct / len(assessed) if assessed else None,
        "comparisons": assessed,
        "note": (
            "This diagnostic compares suggestions with pre-existing human labels. "
            "It must not be generalized to the full collection."
        ),
    }


def evaluate_review_labels(
    input_path: Path = RETROSPECTIVE_PILOT_DIR / "review-labels.csv",
    output_path: Path = REPORTS_DIR / "segmentation-review-evaluation.json",
) -> dict[str, Any]:
    """Validate a reviewer-completed CSV and write an evidence-qualified report."""
    rows = _read_review_rows(input_path)
    classified_by_review = [
        row for row in rows if _normalized(row.get("classification_accepted"))
    ]
    rows_with_new_decisions = [
        row for row in rows
        if _normalized(row.get("classification_accepted"))
        or any(_normalized(row.get(field)) for field in REVIEW_FIELDS)
    ]
    segment_review_documents = sum(
        any(_normalized(row.get(field)) for field in REVIEW_FIELDS)
        for row in rows
    )
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_path),
        "packet_document_count": len(rows),
        "documents_with_new_human_decisions": len(rows_with_new_decisions),
        "new_classification_decision_count": len(classified_by_review),
        "new_classification_decision_counts": dict(sorted(Counter(
            _normalized(row.get("classification_accepted"))
            for row in classified_by_review
        ).items())),
        "existing_label_classification_baseline": _classification_baseline(rows),
        "review_field_summary": _field_review_summary(rows),
        "segmentation_evaluation": {
            "reviewed_documents_with_segment_decisions": segment_review_documents,
            "exhaustive_reference_segments_available": False,
            "precision": None,
            "recall": None,
            "f1": None,
            "status": "insufficient_segment_ground_truth",
            "reason": (
                "The current packet presents one candidate chunk per document and "
                "does not contain exhaustive human reference segment boundaries. "
                "Binary chunk review cannot measure missed segments or valid "
                "segment-level precision, recall, or F1."
            ),
        },
        "ground_truth_mutated": False,
        "registry_mutated": False,
    }

    report["output_json"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    return report


def apply_reviewed_classifications(
    input_path: Path = RETROSPECTIVE_PILOT_DIR / "review-labels.csv",
    registry_path: Path = REGISTRY_FILE,
    output_path: Path = REPORTS_DIR / "classification-review-import.json",
) -> dict[str, Any]:
    """Persist only explicit human-approved classifications from the review CSV."""
    rows = _read_review_rows(input_path)
    registry = load_registry(registry_path)
    records = registry.get("records", {})
    planned: list[dict[str, Any]] = []
    conflicts: list[str] = []

    for row in rows:
        decision = _normalized(row.get("classification_accepted"))
        if not decision:
            continue
        doc_id = row["document_id"]
        record = records.get(doc_id)
        if record is None:
            conflicts.append(f"Document is absent from registry: {doc_id}")
            continue
        if record.get("filename") != row.get("filename"):
            conflicts.append(f"Filename does not match registry for {doc_id}")
            continue

        prior_csv_label = _normalized(row.get("existing_human_classification"))
        current_label = _normalized(record.get("reviewed_classification"))
        if prior_csv_label and prior_csv_label != current_label:
            conflicts.append(f"Human label changed since packet generation for {doc_id}")
            continue

        suggestion = _normalized(row.get("structured_suggested_classification"))
        approved_label = (
            suggestion if decision == "yes"
            else _normalized(row.get("corrected_classification"))
        )
        if approved_label not in CATEGORIES:
            conflicts.append(f"Invalid approved classification for {doc_id}")
            continue
        if current_label and current_label != approved_label:
            conflicts.append(
                f"Refusing to overwrite existing human label for {doc_id}: "
                f"{current_label} -> {approved_label}"
            )
            continue
        planned.append({
            "doc_id": doc_id,
            "filename": row.get("filename", ""),
            "classification": approved_label,
            "decision": "accepted_suggestion" if decision == "yes" else "corrected_suggestion",
            "previous_classification": current_label or None,
            "review_notes": row.get("review_notes", ""),
        })

    if conflicts:
        raise ValueError(
            "Classification import aborted without changes: " + "; ".join(conflicts)
        )

    reviewed_at = datetime.now(timezone.utc).isoformat()
    applied = 0
    unchanged = 0
    for item in planned:
        record = records[item["doc_id"]]
        if record.get("reviewed_classification") == item["classification"]:
            unchanged += 1
            continue
        record["reviewed_classification"] = item["classification"]
        record["classification_review_required"] = False
        record["classification_reviewed_at"] = reviewed_at
        record["classification_review_source"] = input_path.name
        record["classification_review_decision"] = item["decision"]
        if item["review_notes"]:
            record["classification_review_notes"] = item["review_notes"]
        record.setdefault("review", {})["classification"] = "reviewed"
        applied += 1

    if applied:
        save_registry(registry, registry_path)

    report = {
        "schema_version": "1.0",
        "generated_at": reviewed_at,
        "input_csv": str(input_path),
        "registry_path": str(registry_path),
        "classification_rows_considered": len(planned),
        "classifications_applied": applied,
        "already_matching_human_labels": unchanged,
        "registry_mutated": bool(applied),
        "ground_truth_source": "explicit human decisions in review CSV",
        "records": planned,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    report["output_json"] = str(output_path)
    return report
