"""Document inventory and duplicate detection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_SOURCE_DIR,
    REPORTS_DIR,
    SUPPORTED_EXTENSIONS,
)

from .registry import (
    load_registry,
    save_registry,
    upsert_record,
)
from .extractors import assess_pdf_ocr


def sha256_of_file(path: Path) -> str:
    """Calculate SHA-256 without loading the whole file into memory."""

    import hashlib

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def scan_source(
    source_dir: Path,
) -> list[dict[str, Any]]:
    """Scan supported documents recursively."""

    source_dir = source_dir.expanduser().resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        raise NotADirectoryError(
            f"Source directory does not exist: {source_dir}"
        )

    records = []

    for path in sorted(source_dir.rglob("*")):

        if not path.is_file() or path.is_symlink():
            continue

        if path.name.startswith(("~$", ".")):
            continue

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        stat = path.stat()
        sha256 = sha256_of_file(path)

        relative_path = path.relative_to(
            source_dir
        ).as_posix()

        record = {
            "doc_id": sha256,
            "sha256": sha256,
            "filename": path.name,
            "relative_path": relative_path,
            "source_path": str(path),
            "extension": extension,
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
            "mime_type": {
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".odt": "application/vnd.oasis.opendocument.text",
            }[extension],
            "duplicate_status": "pending",
            "suggested_classification": "unknown",
            "classification_confidence": 0.0,
            "extraction_status": "pending",
        }

        records.append(record)

    # Group identical files by content hash.
    groups: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        groups.setdefault(
            record["sha256"],
            [],
        ).append(record)

    for group in groups.values():

        status = (
            "exact_duplicate"
            if len(group) > 1
            else "unique"
        )

        source_files = [
            item["relative_path"]
            for item in group
        ]

        for item in group:
            item["duplicate_status"] = status
            item["source_files"] = source_files

    return records


def _preserve_processing_metadata(
    scanned_record: dict[str, Any],
    existing_record: dict[str, Any],
) -> None:
    """Keep completed extraction metadata when inventory refreshes the same hash."""
    processing_status = existing_record.get("processing", {}).get("status")
    extraction_status = (
        processing_status
        if processing_status in {"pending", "done", "error"}
        else existing_record.get("extraction_status")
    )
    if extraction_status in {"pending", "done", "error"}:
        scanned_record["extraction_status"] = extraction_status

    for field in ("processed_at", "structured_path", "unit_count"):
        if field in existing_record:
            scanned_record[field] = existing_record[field]


def run_inventory(
    source_dir: Path = DEFAULT_SOURCE_DIR,
) -> list[dict[str, Any]]:
    """Scan files and update the central registry."""

    records = scan_source(source_dir)

    registry = load_registry()
    existing_records = registry.get("records", {})

    for record in records:
        doc_id = record["doc_id"]
        current = existing_records.get(doc_id, {})

        # The document ID is its content hash. Re-scanning the same source must
        # not reset completed extraction metadata to the scan default of pending.
        _preserve_processing_metadata(record, current)

        cached_assessment = current.get("ocr_assessment")

        # Non-PDF documents do not require OCR.
        if record["extension"] != ".pdf":
            record["ocr_assessment"] = {
                "required": False,
                "status": "done",
                "method": "not_applicable",
                "assessed_at": None,
            }

        # Reuse the previous assessment if the document has not changed.
        elif (
            isinstance(cached_assessment, dict)
            and cached_assessment.get("status") == "done"
            and isinstance(cached_assessment.get("required"), bool)
        ):
            record["ocr_assessment"] = cached_assessment

        # Assess new PDFs or documents without a valid cached assessment.
        else:
            try:
                record["ocr_assessment"] = assess_pdf_ocr(
                    Path(record["source_path"])
                )

            except Exception as exc:
                record["ocr_assessment"] = {
                    "required": None,
                    "status": "error",
                    "method": "pypdf_text_density_v1",
                    "assessed_at": None,
                    "error": str(exc),
                }

        upsert_record(registry, record)

    save_registry(registry)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = REPORTS_DIR / "inventory.json"
    csv_path = REPORTS_DIR / "inventory.csv"

    json_path.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if records:

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as stream:

            writer = csv.DictWriter(
                stream,
                fieldnames=list(records[0].keys()),
            )

            writer.writeheader()
            writer.writerows(records)

    else:
        csv_path.write_text(
            "",
            encoding="utf-8",
        )

    summary = {
        "source_dir": str(source_dir.resolve()),
        "total_files_supported": len(records),
        "unique_documents_by_content": len({
            record["doc_id"]
            for record in records
        }),
        "files_in_exact_duplicate_groups": sum(
            record["duplicate_status"] == "exact_duplicate"
            for record in records
        ),
    }

    (
        REPORTS_DIR / "inventory-summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return records
