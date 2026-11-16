"""Document processing pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_SOURCE_DIR,
    EXTRACTED_DIR,
    MARKDOWN_DIR,
    STRUCTURED_DIR,
    PILOT_SIZE,
    BATCH_SIZE_DEFAULT,
    OCR_ENABLED,
    OCR_LANGUAGE,
    REGISTRY_FILE,
)
from .extractors import (
    extract_document,
    assess_pdf_ocr,
    OCRRequiredError,
    OCRNotRequiredError,
)
from .registry import load_registry, save_registry, add_history

LOGGER = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_source_path(source_dir: Path, relative_path: str) -> Path:
    root = source_dir.resolve()
    path = (root / relative_path).resolve()

    if path != root and root not in path.parents:
        raise ValueError(f"Invalid path outside source directory: {relative_path}")

    if not path.is_file():
        raise FileNotFoundError(f"Source document not found: {path}")

    return path


def _normalise_units(
    extracted_pages: list[Any],
    extension: str | None = None,
) -> list[dict[str, Any]]:
    """Convert extractor output into consistent text units."""
    units: list[dict[str, Any]] = []
    is_pdf = (extension or "").lower() == ".pdf"

    for position, item in enumerate(extracted_pages, start=1):
        if isinstance(item, dict):
            text = str(item.get("text", "") or "").strip()
            page_number = item.get("page_number", item.get("page"))
        else:
            text = str(item or "").strip()
            page_number = position if is_pdf else None

        if not text:
            continue

        units.append({
            "unit_id": f"unit-{len(units) + 1:06d}",
            "unit_type": "page" if page_number is not None else "document_section",
            "page_number": page_number,
            "text": text,
        })

    return units


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_payload = sanitize_unicode(payload)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            clean_payload,
            stream,
            ensure_ascii=False,
            indent=2,
        )


def _write_markdown(path: Path, record: dict[str, Any], units: list[dict[str, Any]]) -> None:
    lines = [
        f"# {record.get('filename', 'Untitled document')}",
        "",
        f"- Document ID: `{record['doc_id']}`",
        f"- Original file: `{record.get('relative_path', '')}`",
        f"- SHA-256: `{record.get('sha256', '')}`",
        f"- Classification: `{record.get('suggested_classification', 'unclassified')}`",
        "",
        "---",
        "",
    ]

    for unit in units:
        page = unit.get("page_number")
        heading = f"## Page {page}" if page is not None else "## Extracted text"
        lines.extend([heading, "", unit["text"], ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    clean_content = sanitize_unicode("\n".join(lines))
    path.write_text(
        clean_content,
        encoding="utf-8",
    )


def _set_processing(
    registry: dict[str, Any],
    doc_id: str,
    status: str,
    details: str | None = None,
) -> None:
    records = registry.get("records", {})
    record = records.get(doc_id)

    if record is None:
        raise KeyError(f"Document not found in registry: {doc_id}")    
    add_history(
        record,
        "extraction",
        status,
        details,
    )


def select_pilot_set(records: list[dict[str, Any]], limit: int = PILOT_SIZE) -> list[dict[str, Any]]:
    """Select a small, reasonably diverse sample by extension and classification."""
    candidates = [
        record for record in records
        if record.get("processing", {}).get("status") not in {"done"}
    ]

    selected: list[dict[str, Any]] = []
    remaining = candidates.copy()

    while remaining and len(selected) < limit:
        seen: set[tuple[str, str]] = set()

        for record in remaining[:]:
            key = (
                record.get("extension", ""),
                record.get("suggested_classification", "unclassified"),
            )
            if key not in seen:
                selected.append(record)
                remaining.remove(record)
                seen.add(key)

                if len(selected) >= limit:
                    break

    if len(selected) < limit:
        selected_ids = {item["doc_id"] for item in selected}
        selected.extend(
            item for item in remaining
            if item["doc_id"] not in selected_ids
        )
        selected = selected[:limit]

    return selected

def process_document(
    record: dict[str, Any],
    source_dir: Path,
    registry: dict[str, Any],
    ocr_mode: str = "auto",
) -> bool | None:
    doc_id = record["doc_id"]

    try:
        source_path = _safe_source_path(
            source_dir,
            record["relative_path"],
        )

        actual_hash = _sha256(source_path)

        if actual_hash != record.get("sha256"):
            raise ValueError(
                "The source file hash no longer matches the inventory."
            )

        ocr_required = None

        if record.get("extension") == ".pdf":
            assessment = record.get("ocr_assessment", {})

            # Assess the document if no valid cached result exists.
            if (
                assessment.get("status") != "done"
                or not isinstance(assessment.get("required"), bool)
            ):
                assessment = assess_pdf_ocr(source_path)
                record["ocr_assessment"] = assessment

            ocr_required = assessment["required"]

        extracted_pages, extraction_metadata = extract_document(
            source_path,
            ocr_enabled=OCR_ENABLED,
            language=OCR_LANGUAGE,
            ocr_mode=ocr_mode,
            ocr_required=ocr_required,
        )

        units = _normalise_units(
            extracted_pages,
            record.get("extension"),
        )

        if not units:
            raise ValueError(
                "No text could be extracted from this document."
            )

        payload = {
            "doc_id": doc_id,
            "source": {
                "filename": record.get("filename"),
                "relative_path": record.get("relative_path"),
                "sha256": actual_hash,
                "extension": record.get("extension"),
            },
            "extraction": {
                "processed_at": _now(),
                **(extraction_metadata or {}),
            },
            "units": units,
        }

        extracted_path = EXTRACTED_DIR / f"{doc_id}.json"
        markdown_path = MARKDOWN_DIR / f"{doc_id}.md"
        structured_path = STRUCTURED_DIR / f"{doc_id}.json"

        _write_json(extracted_path, payload)
        _write_markdown(markdown_path, record, units)

        structured_payload = {
            "schema_version": "1.0",
            "doc_id": doc_id,
            "source": payload["source"],
            "classification": {
                "suggested": record.get("suggested_classification"),
                "confidence": record.get("classification_confidence"),
                "requires_human_review": record.get(
                    "classification_review_required",
                    True,
                ),
                "reviewed_classification": record.get("reviewed_classification"),
                "review_status": (
                    "reviewed"
                    if record.get("reviewed_classification")
                    else "pending_review"
                ),
                "reviewed_at": record.get("classification_reviewed_at"),
                "reviewed_by": record.get("classification_reviewed_by"),
            },
            "legal_verification": {
                "status": "not_verified",
                "note": (
                    "Document processing does not establish "
                    "legal validity or current force."
                ),
            },
            "units": units,
            "artifacts": {
                "extracted_json": str(extracted_path),
                "markdown": str(markdown_path),
            },
        }

        _write_json(structured_path, structured_payload)

        record["extraction_status"] = "done"
        record["processed_at"] = _now()
        record["structured_path"] = str(structured_path)
        record["unit_count"] = len(units)

        _set_processing(
            registry,
            doc_id,
            "done",
            f"{len(units)} text units extracted.",
        )

        return True

    except (OCRRequiredError, OCRNotRequiredError) as exc:
        LOGGER.info(
            "Document %s deferred: %s",
            doc_id,
            str(exc),
        )

        return None

    except Exception as exc:
        LOGGER.exception(
            "Failed to process document %s",
            doc_id,
        )

        record["extraction_status"] = "error"
        record["processing_error"] = str(exc)

        _set_processing(
            registry,
            doc_id,
            "error",
            str(exc),
        )

        return False
    
# def process_document(
#     record: dict[str, Any],
#     source_dir: Path,
#     registry: dict[str, Any],
# ) -> bool:
#     doc_id = record["doc_id"]

#     try:
#         source_path = _safe_source_path(source_dir, record["relative_path"])
#         actual_hash = _sha256(source_path)

#         if actual_hash != record.get("sha256"):
#             raise ValueError("The source file hash no longer matches the inventory.")

#         extracted_pages, extraction_metadata = extract_document(
#             source_path,
#             ocr_enabled=OCR_ENABLED,
#             language=OCR_LANGUAGE,
#         )

#         units = _normalise_units(extracted_pages, record.get("extension"))
#         if not units:
#             raise ValueError("No text could be extracted from this document.")

#         payload = {
#             "doc_id": doc_id,
#             "source": {
#                 "filename": record.get("filename"),
#                 "relative_path": record.get("relative_path"),
#                 "sha256": actual_hash,
#                 "extension": record.get("extension"),
#             },
#             "extraction": {
#                 "processed_at": _now(),
#                 **(extraction_metadata or {}),
#             },
#             "units": units,
#         }

#         extracted_path = EXTRACTED_DIR / f"{doc_id}.json"
#         markdown_path = MARKDOWN_DIR / f"{doc_id}.md"
#         structured_path = STRUCTURED_DIR / f"{doc_id}.json"

#         _write_json(extracted_path, payload)
#         _write_markdown(markdown_path, record, units)

#         structured_payload = {
#             "schema_version": "1.0",
#             "doc_id": doc_id,
#             "source": payload["source"],
#             "classification": {
#                 "suggested": record.get("suggested_classification"),
#                 "confidence": record.get("classification_confidence"),
#                 "requires_human_review": record.get("classification_review_required", True),
#             },
#             "legal_verification": {
#                 "status": "not_verified",
#                 "note": "Document processing does not establish legal validity or current force.",
#             },
#             "units": units,
#             "artifacts": {
#                 "extracted_json": str(extracted_path),
#                 "markdown": str(markdown_path),
#             },
#         }
#         _write_json(structured_path, structured_payload)

#         record["extraction_status"] = "done"
#         record["processed_at"] = _now()
#         record["structured_path"] = str(structured_path)
#         record["unit_count"] = len(units)
#         _set_processing(registry, doc_id, "done", f"{len(units)} text units extracted.")
#         return True

#     except Exception as exc:
#         LOGGER.exception("Failed to process document %s", doc_id)
#         record["extraction_status"] = "error"
#         record["processing_error"] = str(exc)
#         _set_processing(registry, doc_id, "error", str(exc))
#         return False

def run_processing(
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    pilot: bool = False,
    limit: int = PILOT_SIZE,
    batch_size: int = BATCH_SIZE_DEFAULT,
    retry_errors: bool = False,
    skip_ocr: bool = False,
    ocr_only: bool = False,
) -> dict[str, int]:

    source_dir = Path(source_dir)

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    if skip_ocr and ocr_only:
        raise ValueError(
            "--skip-ocr and --ocr-only cannot be used together."
        )

    if ocr_only and not OCR_ENABLED:
        raise ValueError(
            "--ocr-only requires OCR_ENABLED=true."
        )

    ocr_mode = (
        "skip" if skip_ocr
        else "only" if ocr_only
        else "auto"
    )

    registry = load_registry(REGISTRY_FILE)
    records = list(registry.get("records", {}).values())

    candidates = []

    for record in records:
        status = record.get(
            "processing", {}
        ).get("status", "pending")

        if retry_errors:
            if status != "error":
                continue
        else:
            if status != "pending":
                continue

        candidates.append(record)

    if pilot:
        candidates = select_pilot_set(
            candidates,
            limit,
        )

    counts = {
        "processed": 0,
        "failed": 0,
        "deferred": 0,
        "skipped": 0,
    }

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]

        for record in batch:
            result = process_document(
                record,
                source_dir,
                registry,
                ocr_mode=ocr_mode,
            )

            if result is True:
                counts["processed"] += 1

            elif result is False:
                counts["failed"] += 1

            else:
                counts["deferred"] += 1

        save_registry(
            registry,
            REGISTRY_FILE,
        )

    counts["skipped"] = len(records) - len(candidates)

    return counts

# def run_processing(
#     source_dir: Path | str = DEFAULT_SOURCE_DIR,
#     pilot: bool = False,
#     limit: int = PILOT_SIZE,
#     batch_size: int = BATCH_SIZE_DEFAULT,
#     retry_errors: bool = False,
# ) -> dict[str, int]:
#     source_dir = Path(source_dir)

#     if batch_size < 1:
#         raise ValueError("batch_size must be at least 1.")

#     if limit < 1:
#         raise ValueError("limit must be at least 1.")

#     registry = load_registry(REGISTRY_FILE)
#     records = list(registry.get("records", {}).values())

#     candidates = []

#     for record in records:
#         status = record.get("processing", {}).get("status", "pending")

#         if retry_errors:
#             # Retry only documents previously marked as errors.
#             if status != "error":
#                 continue
#         else:
#             # Normal processing only handles pending documents.
#             if status != "pending":
#                 continue

#         candidates.append(record)

#     if pilot:
#         candidates = select_pilot_set(candidates, limit)

#     counts = {
#         "processed": 0,
#         "failed": 0,
#         "skipped": 0,
#     }

#     for start in range(0, len(candidates), batch_size):
#         batch = candidates[start:start + batch_size]

#         for record in batch:
#             if process_document(record, source_dir, registry):
#                 counts["processed"] += 1
#             else:
#                 counts["failed"] += 1

#         save_registry(registry, REGISTRY_FILE)

#     counts["skipped"] = len(records) - len(candidates)

#     return counts

def validate_outputs() -> dict[str, Any]:
    registry = load_registry(REGISTRY_FILE)
    records = list(registry.get("records", {}).values())

    errors: list[dict[str, str]] = []
    checked = 0

    for record in records:
        if record.get("processing", {}).get("status") != "done":
            continue

        checked += 1
        doc_id = record["doc_id"]
        structured_path = STRUCTURED_DIR / f"{doc_id}.json"

        if not structured_path.is_file():
            errors.append({"doc_id": doc_id, "error": "Structured JSON file is missing."})
            continue

        try:
            payload = json.loads(structured_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append({"doc_id": doc_id, "error": f"Invalid JSON: {exc}"})
            continue

        if payload.get("doc_id") != doc_id:
            errors.append({"doc_id": doc_id, "error": "Document ID mismatch."})

        if not payload.get("units"):
            errors.append({"doc_id": doc_id, "error": "No extracted text units."})

        if payload.get("legal_verification", {}).get("status") != "not_verified":
            errors.append({
                "doc_id": doc_id,
                "error": "Legal verification must remain not_verified until reviewed.",
            })

    return {
        "checked_documents": checked,
        "error_count": len(errors),
        "errors": errors,
        "valid": not errors,
    }
    
def sanitize_unicode(value):
    if isinstance(value, str):
        return value.encode(
            "utf-8",
            errors="replace",
        ).decode("utf-8")

    if isinstance(value, list):
        return [
            sanitize_unicode(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            sanitize_unicode(key): sanitize_unicode(item)
            for key, item in value.items()
        }

    return value
