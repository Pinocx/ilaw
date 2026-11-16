"""Generate a reproducible operational quality report from existing artifacts."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pypdf import PdfReader

from .config import (
    CANDIDATE_INDEX_DIR,
    DEFAULT_SOURCE_DIR,
    MARKDOWN_DIR,
    REGISTRY_FILE,
    REPORTS_DIR,
    STRUCTURED_DIR,
    SUPPORTED_EXTENSIONS,
)
from .registry import load_registry
from .schemas import ChunkMetadata, DocumentMetadata


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _compare_pdf_text_layer(
    source_path: Path,
    structured_units: list[dict[str, Any]],
    markdown_text: str,
) -> dict[str, Any]:
    """Compare replacement characters in stored artifacts with affected PDF text pages."""
    structured_by_page: Counter[int] = Counter()
    for unit in structured_units:
        page = unit.get("page_number")
        count = str(unit.get("text", "")).count("\ufffd")
        if isinstance(page, int) and count:
            structured_by_page[page] += count

    markdown_by_page: Counter[int] = Counter()
    sections = re.split(
        r"(?=^##\s+Page\s+\d+\s*$)",
        markdown_text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    for section in sections:
        match = re.match(
            r"^##\s+Page\s+(\d+)\s*$",
            section,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        if match:
            markdown_by_page[int(match.group(1))] += section[match.end():].count("\ufffd")

    pages = sorted(set(structured_by_page) | set(markdown_by_page))
    if not pages:
        return {
            "status": "not_comparable",
            "reason": "No affected PDF page numbers were present in structured units.",
            "affected_pages": [],
        }

    try:
        reader = PdfReader(str(source_path), strict=False)
        page_results: list[dict[str, Any]] = []
        for page_number in pages:
            if page_number < 1 or page_number > len(reader.pages):
                page_results.append({
                    "page_number": page_number,
                    "structured_count": structured_by_page[page_number],
                    "markdown_count": markdown_by_page[page_number],
                    "pdf_text_layer_count": None,
                    "matches": False,
                })
                continue
            source_text = reader.pages[page_number - 1].extract_text() or ""
            pdf_count = source_text.count("\ufffd")
            page_results.append({
                "page_number": page_number,
                "structured_count": structured_by_page[page_number],
                "markdown_count": markdown_by_page[page_number],
                "pdf_text_layer_count": pdf_count,
                "matches": (
                    structured_by_page[page_number]
                    == markdown_by_page[page_number]
                    == pdf_count
                ),
            })
    except Exception as exc:
        return {
            "status": "error",
            "reason": str(exc)[:500],
            "affected_pages": [],
        }

    status = (
        "matches_pdf_text_layer"
        if all(item["matches"] for item in page_results)
        else "differs_from_pdf_text_layer"
    )
    return {
        "status": status,
        "affected_page_count": len(page_results),
        "replacement_character_count": sum(structured_by_page.values()),
        "affected_pages": page_results,
        "note": (
            "A match confirms preservation of the PDF's extracted text layer; "
            "it does not confirm the visual page content or legal correctness."
        ),
    }


def _scan_source_files(source_dir: Path) -> dict[str, Any]:
    if not source_dir.is_dir():
        return {"directory_exists": False, "files_seen": 0, "unsupported_count": None}

    extensions: Counter[str] = Counter()
    supported_count = 0
    unsupported_examples: list[str] = []
    files_seen = 0
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.name.startswith(("~$", ".")):
            continue
        files_seen += 1
        extension = path.suffix.lower() or "[no extension]"
        extensions[extension] += 1
        if extension in SUPPORTED_EXTENSIONS:
            supported_count += 1
        elif len(unsupported_examples) < 50:
            unsupported_examples.append(path.relative_to(source_dir).as_posix())

    return {
        "directory_exists": True,
        "files_seen": files_seen,
        "supported_files_seen": supported_count,
        "unsupported_count": files_seen - supported_count,
        "files_by_extension": dict(sorted(extensions.items())),
        "unsupported_examples": unsupported_examples,
    }


def _document_findings(
    records: dict[str, dict[str, Any]],
    source_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    processing = Counter()
    extraction = Counter()
    ocr_required = Counter()
    ocr_status = Counter()
    registry_classifications = Counter()
    structured_classifications = Counter()
    reviewed_classifications = Counter()
    classification_discrepancies: list[dict[str, Any]] = []
    missing_artifacts = Counter()
    suspicious_markdown = Counter()
    historical_errors: list[dict[str, Any]] = []
    missing_metadata = Counter()
    pdf_text_layer_statuses = Counter()
    per_document: list[dict[str, Any]] = []
    source_root = source_dir.resolve()
    processing_extraction_status_mismatches = 0

    for doc_id, record in records.items():
        processing_status = record.get("processing", {}).get("status", "pending")
        extraction_status = record.get("extraction_status", "pending")
        processing[processing_status] += 1
        extraction[extraction_status] += 1
        if processing_status != extraction_status:
            processing_extraction_status_mismatches += 1
        ocr = record.get("ocr_assessment", {})
        ocr_required[str(ocr.get("required", "unknown"))] += 1
        ocr_status[str(ocr.get("status", "unknown"))] += 1
        registry_category = record.get("suggested_classification") or "unknown"
        registry_classifications[registry_category] += 1

        relative_path = str(record.get("relative_path", ""))
        source_path = (source_root / relative_path).resolve()
        source_exists = (
            (source_path == source_root or source_root in source_path.parents)
            and source_path.is_file()
        )

        structured_path = STRUCTURED_DIR / f"{doc_id}.json"
        markdown_path = MARKDOWN_DIR / f"{doc_id}.md"
        structured = _read_json(structured_path)
        if not structured_path.is_file():
            missing_artifacts["structured_json"] += 1
        elif not isinstance(structured, dict):
            missing_artifacts["invalid_structured_json"] += 1

        if not markdown_path.is_file():
            missing_artifacts["markdown"] += 1
            markdown_text = ""
        else:
            try:
                markdown_text = markdown_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                markdown_text = ""
                missing_artifacts["unreadable_markdown"] += 1

        if not markdown_text.strip():
            suspicious_markdown["empty_or_unreadable"] += 1
        replacement_count = markdown_text.count("\ufffd")
        if replacement_count:
            suspicious_markdown["replacement_characters_documents"] += 1
            suspicious_markdown["replacement_characters_total"] += replacement_count
        page_heading_count = markdown_text.count("## Page ")
        if record.get("extension", "").lower() == ".pdf" and not page_heading_count:
            suspicious_markdown["pdf_without_page_headings"] += 1

        structured_suggestion = "unknown"
        structured_units = 0
        metadata: dict[str, Any] = {}
        structured_units_data: list[dict[str, Any]] = []
        if isinstance(structured, dict):
            classification = structured.get("classification", {})
            structured_suggestion = classification.get("suggested") or "unknown"
            structured_classifications[structured_suggestion] += 1
            reviewed = (
                record.get("reviewed_classification")
                or classification.get("reviewed_classification")
            )
            if reviewed:
                reviewed_classifications[reviewed] += 1
            units = structured.get("units", [])
            structured_units_data = units if isinstance(units, list) else []
            structured_units = len(structured_units_data)
            metadata = structured.get("document_metadata", structured.get("metadata", {}))
            if not isinstance(metadata, dict):
                metadata = {}
            for field in (
                "jurisdiction", "issuing_authority", "document_number",
                "publication_date", "effective_date", "legal_status",
                "document_version", "reuse_status",
            ):
                if not metadata.get(field):
                    missing_metadata[field] += 1
            if registry_category != structured_suggestion:
                classification_discrepancies.append({
                    "doc_id": doc_id,
                    "filename": record.get("filename"),
                    "registry_suggestion": registry_category,
                    "structured_suggestion": structured_suggestion,
                    "reviewed_classification": reviewed,
                })

        error_events = [
            event for event in record.get("processing", {}).get("history", [])
            if event.get("status") == "error"
        ]
        if error_events:
            historical_errors.append({
                "doc_id": doc_id,
                "filename": record.get("filename"),
                "current_processing_status": processing_status,
                "error_event_count": len(error_events),
                "latest_error": str(error_events[-1].get("details", ""))[:500],
            })

        pdf_text_layer_comparison = None
        if replacement_count and record.get("extension", "").lower() == ".pdf" and source_exists:
            pdf_text_layer_comparison = _compare_pdf_text_layer(
                source_path,
                structured_units_data,
                markdown_text,
            )
            pdf_text_layer_statuses[pdf_text_layer_comparison["status"]] += 1

        per_document.append({
            "doc_id": doc_id,
            "filename": record.get("filename"),
            "source_exists": source_exists,
            "processing_status": processing_status,
            "extraction_status": extraction_status,
            "ocr_required": ocr.get("required"),
            "ocr_status": ocr.get("status"),
            "registry_suggested_classification": registry_category,
            "structured_suggested_classification": structured_suggestion,
            "reviewed_classification": record.get("reviewed_classification"),
            "structured_unit_count": structured_units,
            "markdown_page_heading_count": page_heading_count,
            "markdown_replacement_character_count": replacement_count,
            "pdf_text_layer_comparison": pdf_text_layer_comparison,
            "historical_error_event_count": len(error_events),
        })

    summary = {
        "document_count": len(records),
        "processing_status_counts": dict(processing),
        "extraction_status_counts": dict(extraction),
        "processing_extraction_status_mismatches": processing_extraction_status_mismatches,
        "ocr_required_counts": dict(ocr_required),
        "ocr_status_counts": dict(ocr_status),
        "registry_suggested_classification_counts": dict(registry_classifications),
        "structured_suggested_classification_counts": dict(structured_classifications),
        "reviewed_classification_counts": dict(reviewed_classifications),
        "classification_discrepancy_count": len(classification_discrepancies),
        "missing_artifact_counts": dict(missing_artifacts),
        "suspicious_markdown_counts": dict(suspicious_markdown),
        "historical_error_documents": len(historical_errors),
        "pdf_text_layer_comparison_status_counts": dict(pdf_text_layer_statuses),
        "missing_metadata_document_counts": dict(missing_metadata),
        "source_files_missing": sum(not item["source_exists"] for item in per_document),
    }
    details = {
        "classification_discrepancies": classification_discrepancies,
        "historical_errors": historical_errors,
        "documents": per_document,
    }
    return summary, details


def _candidate_index_findings(index_dir: Path) -> dict[str, Any]:
    report_path = index_dir / "index-report.json"
    report = _read_json(report_path)
    if not isinstance(report, dict):
        return {"available": False, "error": f"Index report missing or invalid: {report_path}"}

    document_names: dict[str, str | None] = {}
    document_metadata_count = 0
    missing_document_metadata = 0
    invalid_document_metadata = 0
    invalid_document_metadata_examples: list[dict[str, Any]] = []
    documents_path = index_dir / "documents.jsonl"
    with documents_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            document = json.loads(line)
            document_names[document["doc_id"]] = document.get("filename")
            payload = document.get("document_metadata")
            if payload is None:
                missing_document_metadata += 1
                continue
            document_metadata_count += 1
            try:
                DocumentMetadata.model_validate(payload)
            except Exception as exc:
                invalid_document_metadata += 1
                if len(invalid_document_metadata_examples) < 10:
                    invalid_document_metadata_examples.append({
                        "line": line_number,
                        "doc_id": document.get("doc_id"),
                        "error": str(exc).splitlines()[0],
                    })

    unit_pages: dict[tuple[str, str], int | None] = {}
    units_path = index_dir / "units.jsonl"
    with units_path.open(encoding="utf-8") as stream:
        for line in stream:
            unit = json.loads(line)
            unit_pages[(unit["doc_id"], unit["unit_id"])] = unit.get("page_number")

    structure_types = Counter()
    chunk_ids: set[str] = set()
    duplicate_chunk_ids = 0
    schema_invalid_chunks = 0
    schema_invalid_examples: list[dict[str, Any]] = []
    missing_document_references = 0
    missing_unit_references = 0
    invalid_page_references = 0
    chunks_without_pages = 0
    chunks_without_pages_by_document: Counter[str] = Counter()
    chunks_without_units = 0
    empty_chunks = 0
    hierarchy_path_mismatches = 0
    chunk_count = 0

    chunks_path = index_dir / "chunks.jsonl"
    with chunks_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            chunk = json.loads(line)
            chunk_count += 1
            chunk_id = chunk.get("chunk_id")
            if chunk_id in chunk_ids:
                duplicate_chunk_ids += 1
            else:
                chunk_ids.add(chunk_id)
            if chunk.get("doc_id") not in document_names:
                missing_document_references += 1
            structure_types[chunk.get("structure_type", "unknown")] += 1
            if not str(chunk.get("text", "")).strip():
                empty_chunks += 1
            refs = chunk.get("source_unit_ids") or []
            if not refs:
                chunks_without_units += 1
            resolved_pages: set[int] = set()
            for ref in refs:
                key = (chunk.get("doc_id"), ref)
                if key not in unit_pages:
                    missing_unit_references += 1
                elif unit_pages[key] is not None:
                    resolved_pages.add(unit_pages[key])
            pages = chunk.get("source_pages") or []
            if not pages:
                chunks_without_pages += 1
                chunks_without_pages_by_document[str(chunk.get("doc_id"))] += 1
            if not set(pages).issubset(resolved_pages):
                invalid_page_references += 1
            hierarchy = chunk.get("hierarchy_path") or []
            reference = chunk.get("structure_reference")
            if reference and hierarchy and hierarchy[-1] != reference:
                hierarchy_path_mismatches += 1

            metadata = {
                key: chunk[key]
                for key in (
                    "chunk_id", "doc_id", "structure_type", "structure_reference",
                    "parent_reference", "hierarchy_path", "source_pages",
                    "source_unit_ids", "start_page", "end_page", "part_number",
                    "part_count", "suggested_classification", "reviewed_classification",
                    "classification_status", "segmentation_confidence",
                    "legal_verification_status", "chunking_method",
                )
                if key in chunk
            }
            try:
                ChunkMetadata.model_validate(metadata)
            except Exception as exc:
                schema_invalid_chunks += 1
                if len(schema_invalid_examples) < 10:
                    schema_invalid_examples.append({
                        "line": line_number,
                        "chunk_id": chunk_id,
                        "error": str(exc).splitlines()[0],
                    })

    coverage = report.get("source_text_coverage", {})
    return {
        "available": True,
        "documents_indexed": report.get("documents_indexed"),
        "documents_with_metadata_schema": document_metadata_count,
        "documents_missing_metadata_schema": missing_document_metadata,
        "invalid_document_metadata": invalid_document_metadata,
        "invalid_document_metadata_examples": invalid_document_metadata_examples,
        "units_written": report.get("units_written"),
        "chunks_written": report.get("chunks_written"),
        "documents_skipped": report.get("documents_skipped"),
        "chunks_read": chunk_count,
        "unique_chunk_ids": len(chunk_ids),
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "schema_invalid_chunks": schema_invalid_chunks,
        "schema_invalid_examples": schema_invalid_examples,
        "missing_document_references": missing_document_references,
        "missing_unit_references": missing_unit_references,
        "invalid_page_references": invalid_page_references,
        "chunks_without_page_references": chunks_without_pages,
        "chunks_without_page_references_by_document": [
            {
                "doc_id": doc_id,
                "filename": document_names.get(doc_id),
                "chunk_count": count,
            }
            for doc_id, count in chunks_without_pages_by_document.most_common()
        ],
        "chunks_without_unit_references": chunks_without_units,
        "empty_chunks": empty_chunks,
        "hierarchy_path_mismatches": hierarchy_path_mismatches,
        "structure_type_counts": dict(structure_types),
        "source_text_coverage": {
            key: value for key, value in coverage.items() if key != "documents"
        },
    }


def build_quality_report(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    index_dir: Path = CANDIDATE_INDEX_DIR,
    json_output: Path | None = None,
    summary_output: Path | None = None,
) -> dict[str, Any]:
    """Inspect existing artifacts and write a report without reprocessing documents."""
    registry = load_registry(REGISTRY_FILE)
    records = registry.get("records", {})
    source_files = _scan_source_files(Path(source_dir))
    inventory_data = _read_json(REPORTS_DIR / "inventory.json")
    inventory_rows = inventory_data if isinstance(inventory_data, list) else []
    duplicate_rows = [
        item for item in inventory_rows
        if item.get("duplicate_status") == "exact_duplicate"
    ]
    duplicate_groups: dict[str, int] = defaultdict(int)
    for item in duplicate_rows:
        duplicate_groups[str(item.get("sha256", "unknown"))] += 1

    document_summary, document_details = _document_findings(records, Path(source_dir))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(Path(source_dir).resolve()),
        "index_dir": str(Path(index_dir).resolve()),
        "inventory": {
            "supported_file_entries": len(inventory_rows),
            "unique_content_hashes": len({item.get("sha256") for item in inventory_rows}),
            "files_in_exact_duplicate_groups": len(duplicate_rows),
            "exact_duplicate_group_count": len(duplicate_groups),
            "source_directory_scan": source_files,
        },
        "documents": document_summary,
        "candidate_index": _candidate_index_findings(Path(index_dir)),
        "details": document_details,
        "interpretation": (
            "Technical integrity and text coverage do not establish legal accuracy, "
            "classification correctness, or source-text fidelity to the original PDF."
        ),
    }

    json_output = json_output or REPORTS_DIR / "repository-quality-report.json"
    summary_output = summary_output or REPORTS_DIR / "inventory-summary.md"
    _atomic_write_json(json_output, report)
    _write_inventory_summary(report, summary_output)
    return {
        "report": str(json_output),
        "inventory_summary": str(summary_output),
        "documents": document_summary["document_count"],
        "chunks": report["candidate_index"].get("chunks_read"),
        "classification_discrepancies": document_summary["classification_discrepancy_count"],
    }


def _write_inventory_summary(report: dict[str, Any], path: Path) -> None:
    inventory = report["inventory"]
    documents = report["documents"]
    candidate = report["candidate_index"]
    coverage = candidate.get("source_text_coverage", {})
    lines = [
        "# Resumen del inventario y calidad técnica",
        "",
        f"Generado: {report['generated_at']}",
        "",
        "## Inventario",
        "",
        f"- Entradas de archivos soportados: {inventory['supported_file_entries']}",
        f"- Documentos únicos por hash: {inventory['unique_content_hashes']}",
        f"- Archivos en grupos duplicados exactos: {inventory['files_in_exact_duplicate_groups']}",
        f"- Archivos con formato no soportado en la carpeta de origen: {inventory['source_directory_scan'].get('unsupported_count', 'no disponible')}",
        "",
        "## Procesamiento y clasificación",
        "",
        f"- Documentos en el registro: {documents['document_count']}",
        f"- Estados de procesamiento: `{json.dumps(documents['processing_status_counts'], ensure_ascii=False)}`",
        f"- Estados de extracción guardados: `{json.dumps(documents['extraction_status_counts'], ensure_ascii=False)}`",
        f"- Diferencias entre propuesta del registro y JSON estructurado: {documents['classification_discrepancy_count']}",
        f"- Documentos con errores históricos de procesamiento: {documents['historical_error_documents']}",
        f"- Comparación de U+FFFD contra la capa de texto PDF: `{json.dumps(documents['pdf_text_layer_comparison_status_counts'], ensure_ascii=False)}`",
        f"- Clasificaciones pendientes de revisión: {sum(documents['processing_status_counts'].values()) - sum(documents['reviewed_classification_counts'].values())}",
        f"- Conteo de documentos marcados OCR-requerido: `{json.dumps(documents['ocr_required_counts'], ensure_ascii=False)}`",
        f"- Estados OCR registrados: `{json.dumps(documents['ocr_status_counts'], ensure_ascii=False)}`",
        "",
        "## Índice candidato",
        "",
        f"- Documentos: {candidate.get('documents_indexed', 'no disponible')}",
        f"- Unidades: {candidate.get('units_written', 'no disponible')}",
        f"- Chunks: {candidate.get('chunks_read', 'no disponible')}",
        f"- Cobertura ordenada de texto no blanco: {coverage.get('coverage_ratio', 'no disponible')}",
        f"- Caracteres de origen reconstruidos: {coverage.get('matched_source_characters', 'no disponible')}",
        f"- Referencias a documento/unidad/página inválidas: {candidate.get('missing_document_references', 0) + candidate.get('missing_unit_references', 0) + candidate.get('invalid_page_references', 0)}",
        f"- Chunks sin referencia de página: {candidate.get('chunks_without_page_references', 'no disponible')}",
        f"- Documentos afectados: `{json.dumps(candidate.get('chunks_without_page_references_by_document', []), ensure_ascii=False)}`",
        "",
        "## Advertencia de interpretación",
        "",
        "Estos indicadores comprueban integridad técnica y preservación de texto extraído. La comparación PDF verifica la capa de texto, no la representación visual de la página. Ninguno de estos controles demuestra que la clasificación, la segmentación jurídica, el OCR o el contenido legal sean correctos.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
