"""Prepare backend-independent JSONL files for future indexing."""

from __future__ import annotations

import json
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .config import CANDIDATE_INDEX_DIR, MARKDOWN_DIR, REGISTRY_FILE
from .registry import load_registry
from .schemas import DocumentMetadata, MetadataFieldProvenance
from .segmentation import segment_document


PAGE_HEADING_RE = re.compile(r"^##\s+Page\s+(\d+)\s*$", re.IGNORECASE)


def _build_document_metadata(
    record: dict[str, Any],
    *,
    reviewed_classification: str | None,
    classification_status: str,
) -> dict[str, Any]:
    """Build official document metadata under the conservative promotion policy."""
    doc_id = str(record.get("doc_id") or "")
    provenance: dict[str, MetadataFieldProvenance] = {}
    source_fields = {
        "source_file": "filename",
        "source_relative_path": "relative_path",
        "source_sha256": "sha256",
        "source_format": "extension",
        "source_size_bytes": "size_bytes",
    }
    values: dict[str, Any] = {}

    for metadata_field, registry_field in source_fields.items():
        value = record.get(registry_field)
        values[metadata_field] = value
        has_value = value is not None
        provenance[metadata_field] = MetadataFieldProvenance(
            status="system_verified" if has_value else "not_established",
            source=f"registry.{registry_field}",
            evidence_ref=(
                f"data/registry.json#/records/{doc_id}/{registry_field}"
                if has_value else None
            ),
        )

    if reviewed_classification:
        provenance["document_type"] = MetadataFieldProvenance(
            status="human_verified",
            source="registry.reviewed_classification",
            evidence_ref=f"data/registry.json#/records/{doc_id}/reviewed_classification",
            reviewed_at=record.get("classification_reviewed_at"),
            reviewed_by=record.get("classification_reviewed_by"),
            review_decision=record.get("classification_review_decision"),
        )
    else:
        provenance["document_type"] = MetadataFieldProvenance(
            status="pending_review",
            source="registry.suggested_classification",
            evidence_ref=f"data/registry.json#/records/{doc_id}/suggested_classification",
        )

    for field_name in (
        "document_title",
        "jurisdiction",
        "issuing_authority",
        "document_number",
        "publication_date",
        "effective_date",
        "legal_status",
    ):
        provenance[field_name] = MetadataFieldProvenance(
            status="not_established",
            source="unverified_source_candidate_or_missing_evidence",
        )

    metadata = DocumentMetadata(
        doc_id=doc_id,
        **values,
        document_type=reviewed_classification,
        classification_status=classification_status,
        field_provenance=provenance,
    )
    return metadata.model_dump(mode="json")


def _markdown_units(markdown: str, doc_id: str) -> list[dict[str, Any]]:
    """Read page-delimited Markdown into the unit format used by segmenter."""
    units: list[dict[str, Any]] = []
    page_number: int | None = None
    lines: list[str] = []

    def flush() -> None:
        text = "\n".join(lines).strip()
        if text:
            units.append({
                "unit_id": f"{doc_id}:page-{page_number or len(units) + 1:06d}",
                "unit_type": "page",
                "page_number": page_number,
                "text": text,
            })

    for line in markdown.splitlines():
        match = PAGE_HEADING_RE.match(line.strip())
        if match:
            flush()
            lines = []
            page_number = int(match.group(1))
        elif page_number is not None:
            lines.append(line)

    flush()

    # Support Markdown files without the page headings produced by this project.
    if not units:
        body = markdown.split("---", 1)[-1].strip()
        if body:
            units.append({
                "unit_id": f"{doc_id}:page-000001",
                "unit_type": "document",
                "page_number": None,
                "text": body,
            })
    return units


def _coverage_metrics(source_text: str, chunk_texts: list[str]) -> dict[str, Any]:
    """Measure ordered source-text representation using non-whitespace tokens."""
    source_tokens = re.findall(r"\S+", source_text)
    chunk_tokens = re.findall(r"\S+", "\n".join(chunk_texts))
    source_characters = sum(map(len, source_tokens))
    chunk_characters = sum(map(len, chunk_tokens))

    if source_tokens == chunk_tokens:
        matched_characters = source_characters
        exact_reconstruction = True
    elif source_tokens and chunk_tokens:
        matcher = SequenceMatcher(None, source_tokens, chunk_tokens)
        matched_characters = sum(
            sum(map(len, source_tokens[source_start:source_start + length]))
            for source_start, _, length in matcher.get_matching_blocks()
        )
        exact_reconstruction = False
    else:
        matched_characters = 0
        exact_reconstruction = not source_tokens and not chunk_tokens

    return {
        "source_non_whitespace_characters": source_characters,
        "chunk_non_whitespace_characters": chunk_characters,
        "matched_source_characters": matched_characters,
        "coverage_ratio": (
            matched_characters / source_characters if source_characters else 1.0
        ),
        "exact_reconstruction": exact_reconstruction,
    }


def build_index_dataset(
    max_chars: int = 1800,
    output_dir: Path = CANDIDATE_INDEX_DIR,
) -> dict[str, int]:
    registry = load_registry(REGISTRY_FILE)
    records = list(registry.get("records", {}).values())
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "documents": output_dir / "documents.jsonl",
        "units": output_dir / "units.jsonl",
        "chunks": output_dir / "chunks.jsonl",
    }
    temporary_paths = {
        name: path.with_suffix(path.suffix + ".tmp")
        for name, path in output_paths.items()
    }

    documents_written = 0
    units_written = 0
    chunks_written = 0
    skipped = 0
    coverage_by_document: list[dict[str, Any]] = []

    try:
        with (
            temporary_paths["documents"].open("w", encoding="utf-8") as documents_stream,
            temporary_paths["units"].open("w", encoding="utf-8") as units_stream,
            temporary_paths["chunks"].open("w", encoding="utf-8") as chunks_stream,
        ):
            for record in records:
                if record.get("processing", {}).get("status") != "done":
                    skipped += 1
                    continue

                doc_id = record["doc_id"]
                markdown_path = MARKDOWN_DIR / f"{doc_id}.md"

                if not markdown_path.is_file():
                    skipped += 1
                    continue

                try:
                    markdown = markdown_path.read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    skipped += 1
                    continue

                units = _markdown_units(markdown, doc_id)
                if not units:
                    skipped += 1
                    continue

                structured_path = MARKDOWN_DIR.parent / "structured" / f"{doc_id}.json"
                try:
                    payload = json.loads(structured_path.read_text(encoding="utf-8"))
                    structured_classification = payload.get("classification", {})
                except (OSError, UnicodeError, json.JSONDecodeError):
                    structured_classification = {}

                suggested_classification = (
                    (
                        record.get("suggested_classification")
                        if record.get("suggested_classification") not in (None, "unknown")
                        else None
                    )
                    or structured_classification.get("suggested")
                    or record.get("suggested_classification")
                    or "unknown"
                )
                classification_confidence = (
                    record.get("classification_confidence")
                    if (
                        record.get("suggested_classification") == suggested_classification
                        and record.get("classification_confidence") is not None
                    )
                    else structured_classification.get("confidence")
                )
                reviewed_classification = (
                    record.get("reviewed_classification")
                    or structured_classification.get("reviewed_classification")
                )
                review_status = (
                    "reviewed"
                    if reviewed_classification
                    else "pending_review"
                )
                # Only human-reviewed categories may control legal article parsing.
                segmentation_classification = {
                    "suggested": reviewed_classification or "unknown"
                }

                document = {
                    "doc_id": doc_id,
                    "filename": record.get("filename"),
                    "relative_path": record.get("relative_path"),
                    "sha256": record.get("sha256"),
                    "extension": record.get("extension"),
                    "suggested_classification": suggested_classification,
                    "reviewed_classification": reviewed_classification,
                    "classification_status": review_status,
                    "classification_confidence": classification_confidence,
                    "classification_requires_human_review": review_status != "reviewed",
                    "legal_verification_status": "not_verified",
                    "unit_count": len(units),
                    "document_metadata": _build_document_metadata(
                        record,
                        reviewed_classification=reviewed_classification,
                        classification_status=review_status,
                    ),
                }
                documents_stream.write(json.dumps(document, ensure_ascii=False) + "\n")
                documents_written += 1

                for unit in units:
                    unit_text = unit.get("text", "").strip()
                    if not unit_text:
                        continue
                    units_stream.write(json.dumps({
                        "doc_id": doc_id,
                        "unit_id": unit.get("unit_id"),
                        "unit_type": unit.get("unit_type"),
                        "page_number": unit.get("page_number"),
                        "text": unit_text,
                        "suggested_classification": suggested_classification,
                        "reviewed_classification": reviewed_classification,
                        "classification_status": review_status,
                        "legal_verification_status": "not_verified",
                        "source": {
                            "filename": record.get("filename"),
                            "relative_path": record.get("relative_path"),
                            "sha256": record.get("sha256"),
                        },
                    }, ensure_ascii=False) + "\n")
                    units_written += 1

                document_chunks: list[str] = []
                for chunk in segment_document(
                    doc_id=doc_id,
                    units=units,
                    classification=segmentation_classification,
                    max_chars=max_chars,
                ):
                    chunk["suggested_classification"] = suggested_classification
                    chunk["reviewed_classification"] = reviewed_classification
                    chunk["classification_status"] = review_status
                    chunks_stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    document_chunks.append(chunk["text"])
                    chunks_written += 1

                coverage = _coverage_metrics(
                    "\n".join(unit["text"] for unit in units),
                    document_chunks,
                )
                coverage_by_document.append({
                    "doc_id": doc_id,
                    "filename": record.get("filename"),
                    **coverage,
                })

        for name, path in output_paths.items():
            # Windows may deny replacing an existing file while another
            # process has it open. Copy only after the complete temp file exists.
            with temporary_paths[name].open("rb") as source, path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            temporary_paths[name].unlink()
    except Exception:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        raise

    report = {
        "documents_indexed": documents_written,
        "units_written": units_written,
        "chunks_written": chunks_written,
        "documents_skipped": skipped,
        "source_text_coverage": {
            "method": "ordered_non_whitespace_token_matching_v1",
            "documents_measured": len(coverage_by_document),
            "source_non_whitespace_characters": sum(
                item["source_non_whitespace_characters"]
                for item in coverage_by_document
            ),
            "matched_source_characters": sum(
                item["matched_source_characters"]
                for item in coverage_by_document
            ),
            "coverage_ratio": (
                sum(item["matched_source_characters"] for item in coverage_by_document)
                / max(1, sum(
                    item["source_non_whitespace_characters"]
                    for item in coverage_by_document
                ))
            ),
            "exact_reconstruction_documents": sum(
                item["exact_reconstruction"] for item in coverage_by_document
            ),
            "documents": coverage_by_document,
        },
        "note": (
            "Chunks were generated from Markdown using adaptive structural rules. "
            "Source coverage is an ordered token-level technical measure, not legal validation. "
            "No embeddings or vector database have been generated."
        ),
    }

    report_path = output_dir / "index-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return report
