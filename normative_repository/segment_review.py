"""Create exhaustive page-review tasks and evaluate human reference segments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    CANDIDATE_INDEX_DIR,
    RETROSPECTIVE_PILOT_DIR,
    SEGMENT_EVALUATION_DIR,
    STRUCTURED_DIR,
    REPORTS_DIR,
)
from .schemas import SegmentType


TOKEN_RE = re.compile(r"\S+")
PAGE_REVIEW_FIELDS = (
    "task_id",
    "document_id",
    "filename",
    "page_number",
    "source_unit_id",
    "page_text_sha256",
    "page_text_fidelity",
    "exhaustive_annotation_complete",
    "review_notes",
)
GOLD_SEGMENT_FIELDS = (
    "task_id",
    "segment_order",
    "segment_text",
    "structure_type",
    "structural_reference",
    "parent_reference",
    "hierarchy_path",
    "review_notes",
)
FIDELITY_VALUES = {"yes", "no", "partial", "not_applicable"}
COMPLETION_VALUES = {"yes", "no"}
SEGMENT_TYPES = {item.value for item in SegmentType}


def _tokens(value: str) -> list[str]:
    return TOKEN_RE.findall(value)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header row: {path}")
            return list(reader.fieldnames), [
                {key: str(value or "").strip() for key, value in row.items() if key}
                for row in reader
            ]
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read CSV: {path}") from exc


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _ensure_no_completed_annotations(output_dir: Path) -> None:
    page_review_path = output_dir / "page-review.csv"
    if page_review_path.is_file():
        _, rows = _read_csv(page_review_path)
        if any(
            row.get("page_text_fidelity") or row.get("exhaustive_annotation_complete")
            or row.get("review_notes")
            for row in rows
        ):
            raise FileExistsError(
                f"Human page-review decisions already exist in {page_review_path}; "
                "refusing to replace them."
            )
    gold_path = output_dir / "gold-segments.csv"
    if gold_path.is_file():
        _, rows = _read_csv(gold_path)
        if any(
            row.get("segment_text") or row.get("structure_type")
            or row.get("review_notes")
            for row in rows
        ):
            raise FileExistsError(
                f"Human segment annotations already exist in {gold_path}; "
                "refusing to replace them."
            )


def _select_tasks(
    review_csv: Path,
    structured_dir: Path,
    pages_per_document: int,
    seed: int,
) -> list[dict[str, Any]]:
    _, reviewed_docs = _read_csv(review_csv)
    if pages_per_document < 1 or pages_per_document > 20:
        raise ValueError("pages_per_document must be between 1 and 20.")

    tasks: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    for review_doc in reviewed_docs:
        doc_id = review_doc.get("document_id", "")
        if not doc_id or doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        source_format = review_doc.get("source_format", "").casefold()
        if source_format not in {".pdf", "pdf"} and Path(source_format).suffix != ".pdf":
            continue
        structured_path = structured_dir / f"{doc_id}.json"
        try:
            payload = json.loads(structured_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load structured source for {doc_id}.") from exc
        units = payload.get("units", [])
        pages = [
            unit for unit in units
            if isinstance(unit.get("page_number"), int)
            and isinstance(unit.get("text"), str)
            and unit["text"].strip()
        ]
        unique_pages: dict[int, dict[str, Any]] = {}
        for unit in pages:
            unique_pages.setdefault(unit["page_number"], unit)
        page_numbers = sorted(unique_pages)
        if not page_numbers:
            continue
        count = min(pages_per_document, len(page_numbers))
        sampler = random.Random(f"{seed}:{doc_id}")
        selected_pages = sorted(sampler.sample(page_numbers, count))
        filename = review_doc.get("filename", payload.get("source", {}).get("filename", ""))
        for page_number in selected_pages:
            unit = unique_pages[page_number]
            text = unit["text"]
            unit_id = str(unit.get("unit_id") or f"{doc_id}:page-{page_number:06d}")
            task_id = f"{doc_id}:page-{page_number:06d}"
            tasks.append({
                "task_id": task_id,
                "document_id": doc_id,
                "filename": filename,
                "page_number": page_number,
                "source_unit_id": unit_id,
                "page_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "page_text": text,
                "token_count": len(_tokens(text)),
            })
    return tasks


def build_segment_review_packet(
    *,
    review_csv: Path = RETROSPECTIVE_PILOT_DIR / "review-labels.csv",
    structured_dir: Path = STRUCTURED_DIR,
    output_dir: Path = SEGMENT_EVALUATION_DIR,
    pages_per_document: int = 2,
    seed: int = 2026,
) -> dict[str, Any]:
    """Build a reproducible source-only packet for exhaustive page annotations."""
    tasks = _select_tasks(review_csv, structured_dir, pages_per_document, seed)
    if not tasks:
        raise ValueError("No page-addressable PDF tasks were found in the review CSV.")
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_no_completed_annotations(output_dir)

    page_review_rows = [
        {
            **{key: task[key] for key in PAGE_REVIEW_FIELDS[:6]},
            "page_text_fidelity": "",
            "exhaustive_annotation_complete": "",
            "review_notes": "",
        }
        for task in tasks
    ]
    gold_rows = [
        {
            "task_id": task["task_id"],
            "segment_order": 1,
            "segment_text": "",
            "structure_type": "",
            "structural_reference": "",
            "parent_reference": "",
            "hierarchy_path": "[]",
            "review_notes": "",
        }
        for task in tasks
    ]
    _write_csv(output_dir / "page-review.csv", PAGE_REVIEW_FIELDS, page_review_rows)
    _write_csv(output_dir / "gold-segments.csv", GOLD_SEGMENT_FIELDS, gold_rows)

    packet_lines = [
        "# Revisión exhaustiva de segmentación por página",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Sampling seed: {seed}; pages per PDF: {pages_per_document}",
        f"Page tasks: {len(tasks)}; development split (not held out).",
        "",
        "## Reviewer instructions",
        "",
        "1. Review the original PDF page and mark `page_text_fidelity` in `page-review.csv`.",
        "2. In `gold-segments.csv`, add one row for every exhaustive, non-overlapping text segment on that page, in source order. Copy the exact segment text; whitespace differences are ignored, but words and punctuation must remain faithful.",
        "3. Label every non-whitespace token. Include ordinary prose as `paragraph` or `other`, table-of-contents entries as `table_of_contents_entry`, and quoted law as `quoted_provision` when appropriate. Do not omit covers, footnotes, or headings; use the matching `SegmentType` value.",
        "   Allowed values: " + ", ".join(f"`{value}`" for value in sorted(SEGMENT_TYPES)) + ".",
        "4. Set `exhaustive_annotation_complete=yes` only after all page text is covered exactly once by the ordered segment rows. Otherwise use `no` and explain the limitation.",
        "5. Set hierarchy paths as JSON arrays from the top-level parent through the segment. Use `[]` when no hierarchy applies.",
        "6. Do not consult candidate chunks while creating gold annotations. The evaluator compares them only after the gold rows are complete.",
        "",
        "The deterministic evaluator checks exact ordered token coverage for each complete page. It reports segment precision, recall, and F1 using one-to-one span matching at token IoU >= 0.5, plus token-level structural-label metrics. These are page-sample metrics, not whole-document metrics.",
        "",
    ]
    for index, task in enumerate(tasks, start=1):
        text = task["page_text"]
        fence = "`" * max(4, max((len(match.group(0)) for match in re.finditer(r"`+", text)), default=0) + 1)
        packet_lines.extend([
            f"## Task {index}: {task['filename']} — page {task['page_number']}",
            "",
            f"- Task ID: `{task['task_id']}`",
            f"- Source unit: `{task['source_unit_id']}`",
            f"- Non-whitespace tokens to annotate: `{task['token_count']}`",
            f"- Text SHA-256: `{task['page_text_sha256']}`",
            "",
            fence + "text",
            text,
            fence,
            "",
        ])
    packet_path = output_dir / "page-review-packet.md"
    packet_path.write_text("\n".join(packet_lines), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "pages_per_document": pages_per_document,
        "split": "development",
        "reviewed_document_count": len({task["document_id"] for task in tasks}),
        "task_count": len(tasks),
        "source_review_csv": str(review_csv),
        "candidate_chunks_shown_to_reviewer": False,
        "tasks": [
            {key: task[key] for key in (
                "task_id", "document_id", "filename", "page_number",
                "source_unit_id", "page_text_sha256", "token_count",
            )}
            for task in tasks
        ],
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "task_count": len(tasks),
        "reviewed_document_count": manifest["reviewed_document_count"],
        "packet": str(packet_path),
        "page_review_csv": str(output_dir / "page-review.csv"),
        "gold_segments_csv": str(output_dir / "gold-segments.csv"),
        "manifest": str(manifest_path),
        "split": "development",
    }


def _load_jsonl_by_document(path: Path, document_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    records = {doc_id: [] for doc_id in document_ids}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from exc
                doc_id = record.get("doc_id")
                if doc_id in records:
                    records[doc_id].append(record)
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read candidate index file: {path}") from exc
    return records


def _group_predicted_segments(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    segments: list[dict[str, Any]] = []
    errors: list[str] = []
    offset = 0
    index = 0
    while index < len(chunks):
        first = chunks[index]
        part_count = int(first.get("part_count", 1))
        if int(first.get("part_number", 1)) != 1 or part_count < 1:
            errors.append(f"Invalid chunk part sequence at {first.get('chunk_id')}.")
            part_count = 1
        start = offset
        parts = chunks[index:index + part_count]
        expected = list(range(1, len(parts) + 1))
        actual = [int(part.get("part_number", 1)) for part in parts]
        if actual != expected or len(parts) != part_count:
            errors.append(f"Incomplete split segment at {first.get('chunk_id')}.")
        for part in parts:
            offset += len(_tokens(str(part.get("text", ""))))
        segments.append({
            "start": start,
            "end": offset,
            "structure_type": first.get("structure_type", "other"),
            "structure_reference": first.get("structure_reference"),
            "parent_reference": first.get("parent_reference"),
            "hierarchy_path": first.get("hierarchy_path") or [],
            "source_pages": first.get("source_pages") or [],
            "source_unit_ids": first.get("source_unit_ids") or [],
            "chunk_ids": [part.get("chunk_id") for part in parts],
        })
        index += len(parts)
    return segments, errors


def _maximum_matching(adjacency: list[list[int]]) -> dict[int, int]:
    matched_predictions: dict[int, int] = {}

    def augment(gold_index: int, visited: set[int]) -> bool:
        for prediction_index in adjacency[gold_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous = matched_predictions.get(prediction_index)
            if previous is None or augment(previous, visited):
                matched_predictions[prediction_index] = gold_index
                return True
        return False

    for gold_index in range(len(adjacency)):
        augment(gold_index, set())
    return {gold_index: prediction_index for prediction_index, gold_index in matched_predictions.items()}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _span_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union else 0.0


def _segment_metrics(
    gold: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    adjacency: list[list[int]] = []
    for gold_segment in gold:
        candidates = []
        gold_span = (gold_segment["start"], gold_segment["end"])
        for index, prediction in enumerate(predicted):
            if prediction["structure_type"] != gold_segment["structure_type"]:
                continue
            iou = _span_iou(gold_span, (prediction["start"], prediction["end"]))
            if iou >= threshold:
                candidates.append((iou, index))
        adjacency.append([index for _, index in sorted(candidates, reverse=True)])
    matches = _maximum_matching(adjacency)
    true_positive = len(matches)
    false_positive = len(predicted) - true_positive
    false_negative = len(gold) - true_positive
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0 if precision is not None and recall is not None else None
    )
    return {
        "true_positive_segments": true_positive,
        "false_positive_segments": false_positive,
        "false_negative_segments": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matching_threshold_token_iou": threshold,
        "matched_pairs": [
            {
                "gold_index": gold_index,
                "prediction_index": prediction_index,
                "token_iou": _span_iou(
                    (gold[gold_index]["start"], gold[gold_index]["end"]),
                    (predicted[prediction_index]["start"], predicted[prediction_index]["end"]),
                ),
            }
            for gold_index, prediction_index in sorted(matches.items())
        ],
    }


def _token_metrics(confusion: Counter[tuple[str, str]]) -> dict[str, Any]:
    labels = sorted({label for pair in confusion for label in pair})
    total = sum(confusion.values())
    correct = sum(confusion[(label, label)] for label in labels)
    per_type: dict[str, Any] = {}
    for label in labels:
        true_positive = confusion[(label, label)]
        false_positive = sum(confusion[(gold, label)] for gold in labels if gold != label)
        false_negative = sum(confusion[(label, predicted)] for predicted in labels if predicted != label)
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0 if precision is not None and recall is not None else None
        )
        per_type[label] = {
            "support_tokens": true_positive + false_negative,
            "predicted_tokens": true_positive + false_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "total_tokens": total,
        "correctly_labeled_tokens": correct,
        "accuracy": _ratio(correct, total),
        "micro_precision": _ratio(correct, total),
        "micro_recall": _ratio(correct, total),
        "micro_f1": _ratio(correct, total),
        "per_type": per_type,
        "confusion_matrix": {
            gold: {predicted: confusion[(gold, predicted)] for predicted in labels}
            for gold in labels
        },
    }


def evaluate_segment_review_packet(
    *,
    packet_dir: Path = SEGMENT_EVALUATION_DIR,
    structured_dir: Path = STRUCTURED_DIR,
    index_dir: Path = CANDIDATE_INDEX_DIR,
    output_path: Path = REPORTS_DIR / "segmentation-page-evaluation.json",
    matching_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate exhaustive page annotations against candidate segments."""
    if not 0 < matching_threshold <= 1:
        raise ValueError("matching_threshold must be greater than 0 and at most 1.")
    manifest_path = packet_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FileNotFoundError(f"Cannot read review manifest: {manifest_path}") from exc

    _, review_rows = _read_csv(packet_dir / "page-review.csv")
    _, gold_rows = _read_csv(packet_dir / "gold-segments.csv")
    tasks = {task["task_id"]: task for task in manifest.get("tasks", [])}
    review_by_task: dict[str, dict[str, str]] = {}
    for row in review_rows:
        task_id = row.get("task_id", "")
        if task_id not in tasks or task_id in review_by_task:
            raise ValueError(f"Unknown or duplicate page-review task: {task_id}")
        fidelity = row.get("page_text_fidelity", "").casefold()
        complete = row.get("exhaustive_annotation_complete", "").casefold()
        if fidelity and fidelity not in FIDELITY_VALUES:
            raise ValueError(f"Invalid page_text_fidelity for task {task_id}.")
        if complete and complete not in COMPLETION_VALUES:
            raise ValueError(f"Invalid exhaustive_annotation_complete for task {task_id}.")
        review_by_task[task_id] = row

    grouped_gold: dict[str, list[dict[str, str]]] = {task_id: [] for task_id in tasks}
    for row in gold_rows:
        task_id = row.get("task_id", "")
        if task_id not in tasks:
            raise ValueError(f"Unknown gold segment task: {task_id}")
        if not row.get("segment_text", "").strip():
            continue
        segment_type = row.get("structure_type", "").casefold()
        if segment_type not in SEGMENT_TYPES:
            raise ValueError(
                f"Invalid structure_type for task {task_id}: {segment_type}"
            )
        try:
            order = int(row.get("segment_order", ""))
        except ValueError as exc:
            raise ValueError(f"Invalid segment_order for task {task_id}.") from exc
        if order < 1:
            raise ValueError(f"segment_order must be positive for task {task_id}.")
        row["_order"] = str(order)
        grouped_gold[task_id].append(row)

    eligible_tasks = [
        task_id for task_id in tasks
        if review_by_task.get(task_id, {}).get("page_text_fidelity", "").casefold() == "yes"
        and review_by_task.get(task_id, {}).get("exhaustive_annotation_complete", "").casefold() == "yes"
    ]
    if not eligible_tasks:
        report = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "packet_manifest": str(manifest_path),
            "evaluation_scope": "Exhaustively annotated pages from the development pilot; not whole-document metrics or a held-out evaluation.",
            "packet_task_count": len(tasks),
            "completed_task_count": sum(
                row.get("exhaustive_annotation_complete", "").casefold() == "yes"
                for row in review_by_task.values()
            ),
            "evaluated_page_count": 0,
            "skipped_pages": [
                {"task_id": task_id, "reason": (
                    "annotation_not_complete"
                    if review_by_task.get(task_id, {}).get("exhaustive_annotation_complete", "").casefold() != "yes"
                    else "source_text_not_confirmed_faithful"
                )}
                for task_id in tasks
            ],
            "candidate_source_alignment_errors": [],
            "segment_level": {
                **_segment_metrics_from_counts(0, 0, 0, matching_threshold),
                "gold_segments": 0,
                "predicted_segments": 0,
                "matched_segments": 0,
                "status": "awaiting_complete_human_annotations",
                "per_page": [],
            },
            "token_level_structure_labels": _token_metrics(Counter()),
            "ground_truth_mutated": False,
            "registry_mutated": False,
        }
        report["output_json"] = str(output_path)
        _write_json(output_path, report)
        return report

    document_ids = {tasks[task_id]["document_id"] for task_id in eligible_tasks}
    source_payloads: dict[str, dict[str, Any]] = {}
    page_offsets: dict[str, dict[int, tuple[int, int, list[str]]]] = {}
    page_texts: dict[str, dict[int, str]] = {}
    source_token_sequences: dict[str, list[str]] = {}
    source_alignment_errors: list[str] = []
    for doc_id in document_ids:
        path = structured_dir / f"{doc_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load structured source for {doc_id}.") from exc
        source_payloads[doc_id] = payload
        units = sorted(
            [
                unit for unit in payload.get("units", [])
                if isinstance(unit.get("page_number"), int)
                and isinstance(unit.get("text"), str)
                and unit["text"].strip()
            ],
            key=lambda unit: unit["page_number"],
        )
        offsets: dict[int, tuple[int, int, list[str]]] = {}
        page_texts[doc_id] = {}
        all_tokens: list[str] = []
        for unit in units:
            page = unit["page_number"]
            unit_tokens = _tokens(unit["text"])
            start = len(all_tokens)
            all_tokens.extend(unit_tokens)
            offsets[page] = (start, len(all_tokens), unit_tokens)
            page_texts[doc_id][page] = unit["text"]
        page_offsets[doc_id] = offsets
        source_token_sequences[doc_id] = all_tokens

    chunks_by_doc = _load_jsonl_by_document(index_dir / "chunks.jsonl", document_ids)
    predicted_by_doc: dict[str, list[dict[str, Any]]] = {}
    for doc_id, chunks in chunks_by_doc.items():
        predicted, errors = _group_predicted_segments(chunks)
        source_tokens = source_token_sequences[doc_id]
        candidate_tokens = [
            token for chunk in chunks for token in _tokens(str(chunk.get("text", "")))
        ]
        if source_tokens != candidate_tokens:
            source_alignment_errors.append(
                f"Candidate tokens do not exactly reconstruct source units for {doc_id}."
            )
        source_alignment_errors.extend(errors)
        predicted_by_doc[doc_id] = predicted

    evaluated_pages: list[dict[str, Any]] = []
    skipped_pages: list[dict[str, str]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    gold_segment_count = 0
    predicted_segment_count = 0
    matched_segment_count = 0
    metadata_checks: dict[str, Counter[str]] = {
        "structural_reference": Counter(),
        "parent_reference": Counter(),
        "hierarchy_path": Counter(),
        "page_reference": Counter(),
    }

    for task_id, task in tasks.items():
        review = review_by_task.get(task_id, {})
        fidelity = review.get("page_text_fidelity", "").casefold()
        complete = review.get("exhaustive_annotation_complete", "").casefold()
        if complete != "yes":
            skipped_pages.append({"task_id": task_id, "reason": "annotation_not_complete"})
            continue
        if fidelity != "yes":
            skipped_pages.append({"task_id": task_id, "reason": "source_text_not_confirmed_faithful"})
            continue
        if source_alignment_errors:
            skipped_pages.append({"task_id": task_id, "reason": "candidate_source_alignment_failed"})
            continue

        doc_id = task["document_id"]
        page = int(task["page_number"])
        if page not in page_offsets[doc_id]:
            raise ValueError(f"Source page disappeared for task {task_id}.")
        page_start, page_end, source_page_tokens = page_offsets[doc_id][page]
        actual_hash = hashlib.sha256(page_texts[doc_id][page].encode("utf-8")).hexdigest()
        if actual_hash != task["page_text_sha256"]:
            raise ValueError(f"Source page changed since packet generation: {task_id}")

        task_gold_rows = sorted(grouped_gold[task_id], key=lambda row: int(row["_order"]))
        orders = [int(row["_order"]) for row in task_gold_rows]
        if len(orders) != len(set(orders)):
            raise ValueError(f"Duplicate segment_order for task {task_id}.")
        gold_token_sequence = [
            token for row in task_gold_rows for token in _tokens(row["segment_text"])
        ]
        if gold_token_sequence != source_page_tokens:
            raise ValueError(
                f"Gold segments for {task_id} do not cover the complete page text "
                "exactly once and in source order."
            )

        gold_spans: list[dict[str, Any]] = []
        local_cursor = 0
        for row in task_gold_rows:
            segment_tokens = _tokens(row["segment_text"])
            end = local_cursor + len(segment_tokens)
            try:
                hierarchy = json.loads(row.get("hierarchy_path") or "[]")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid hierarchy_path for task {task_id}.") from exc
            if not isinstance(hierarchy, list) or not all(isinstance(item, str) for item in hierarchy):
                raise ValueError(f"hierarchy_path must be a JSON string array for task {task_id}.")
            gold_spans.append({
                "start": local_cursor,
                "end": end,
                "structure_type": row["structure_type"].casefold(),
                "structural_reference": row.get("structural_reference") or None,
                "parent_reference": row.get("parent_reference") or None,
                "hierarchy_path": hierarchy,
            })
            local_cursor = end

        predicted_spans: list[dict[str, Any]] = []
        for prediction in predicted_by_doc[doc_id]:
            start = max(page_start, prediction["start"])
            end = min(page_end, prediction["end"])
            if start >= end:
                continue
            predicted_spans.append({
                **prediction,
                "start": start - page_start,
                "end": end - page_start,
            })

        for gold in gold_spans:
            for position in range(gold["start"], gold["end"]):
                predicted_type = next(
                    (
                        item["structure_type"] for item in predicted_spans
                        if item["start"] <= position < item["end"]
                    ),
                    "other",
                )
                confusion[(gold["structure_type"], str(predicted_type))] += 1

        page_segment_metrics = _segment_metrics(
            gold_spans,
            predicted_spans,
            matching_threshold,
        )
        for pair in page_segment_metrics["matched_pairs"]:
            gold = gold_spans[pair["gold_index"]]
            prediction = predicted_spans[pair["prediction_index"]]
            metadata_checks["structural_reference"][
                "correct" if _normalize_reference(gold.get("structural_reference"))
                == _normalize_reference(prediction.get("structure_reference")) else "incorrect"
            ] += 1
            metadata_checks["parent_reference"][
                "correct" if _normalize_reference(gold.get("parent_reference"))
                == _normalize_reference(prediction.get("parent_reference")) else "incorrect"
            ] += 1
            gold_hierarchy = [_normalize_reference(item) for item in gold["hierarchy_path"]]
            predicted_hierarchy = [
                _normalize_reference(item) for item in prediction.get("hierarchy_path", [])
            ]
            metadata_checks["hierarchy_path"][
                "correct" if gold_hierarchy == predicted_hierarchy else "incorrect"
            ] += 1
            metadata_checks["page_reference"][
                "correct" if page in prediction.get("source_pages", []) else "incorrect"
            ] += 1
        gold_segment_count += len(gold_spans)
        predicted_segment_count += len(predicted_spans)
        matched_segment_count += page_segment_metrics["true_positive_segments"]
        evaluated_pages.append({
            "task_id": task_id,
            "document_id": doc_id,
            "filename": task["filename"],
            "page_number": page,
            "token_count": len(source_page_tokens),
            "gold_segment_count": len(gold_spans),
            "predicted_segment_count": len(predicted_spans),
            "segment_metrics": page_segment_metrics,
        })

    overall_segment_metrics = _segment_metrics_from_counts(
        matched_segment_count,
        predicted_segment_count,
        gold_segment_count,
        matching_threshold,
    )
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packet_manifest": str(manifest_path),
        "evaluation_scope": "Exhaustively annotated pages from the development pilot; not whole-document metrics or a held-out evaluation.",
        "packet_task_count": len(tasks),
        "completed_task_count": sum(
            row.get("exhaustive_annotation_complete", "").casefold() == "yes"
            for row in review_by_task.values()
        ),
        "evaluated_page_count": len(evaluated_pages),
        "skipped_pages": skipped_pages,
        "candidate_source_alignment_errors": source_alignment_errors,
        "segment_level": {
            **overall_segment_metrics,
            "gold_segments": gold_segment_count,
            "predicted_segments": predicted_segment_count,
            "matched_segments": matched_segment_count,
            "status": "measured" if evaluated_pages else "awaiting_complete_human_annotations",
            "per_page": evaluated_pages,
        },
        "matched_segment_metadata": {
            field: {
                "correct": counts.get("correct", 0),
                "incorrect": counts.get("incorrect", 0),
                "accuracy": _ratio(
                    counts.get("correct", 0),
                    counts.get("correct", 0) + counts.get("incorrect", 0),
                ),
            }
            for field, counts in metadata_checks.items()
        },
        "token_level_structure_labels": _token_metrics(confusion),
        "ground_truth_mutated": False,
        "registry_mutated": False,
    }
    report["output_json"] = str(output_path)
    _write_json(output_path, report)
    return report


def _segment_metrics_from_counts(
    true_positive: int,
    predicted_count: int,
    gold_count: int,
    threshold: float,
) -> dict[str, Any]:
    false_positive = predicted_count - true_positive
    false_negative = gold_count - true_positive
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0 if precision is not None and recall is not None else None
    )
    return {
        "true_positive_segments": true_positive,
        "false_positive_segments": false_positive,
        "false_negative_segments": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matching_threshold_token_iou": threshold,
    }


def _normalize_reference(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()
