"""Build a source-grounded packet for retrospective human quality review."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import (
    CANDIDATE_INDEX_DIR,
    MARKDOWN_DIR,
    REGISTRY_FILE,
    REPORTS_DIR,
    RETROSPECTIVE_PILOT_DIR,
    STRUCTURED_DIR,
)
from .registry import load_registry


PAGE_HEADING_RE = re.compile(r"^##\s+Page\s+(\d+)\s*$", re.MULTILINE | re.IGNORECASE)


def _read_structured(record: dict[str, Any]) -> dict[str, Any]:
    path = STRUCTURED_DIR / f"{record['doc_id']}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload


def _classification(record: dict[str, Any], payload: dict[str, Any]) -> str:
    structured = payload.get("classification", {})
    return (
        record.get("suggested_classification")
        if record.get("suggested_classification") not in (None, "unknown")
        else structured.get("suggested") or record.get("suggested_classification") or "unknown"
    )


def _choose_records(limit: int) -> list[dict[str, Any]]:
    registry = load_registry(REGISTRY_FILE)
    quality_path = REPORTS_DIR / "repository-quality-report.json"
    replacement_counts: dict[str, int] = {}
    pdf_text_layer_audits: dict[str, dict[str, Any] | None] = {}
    try:
        quality_report = json.loads(quality_path.read_text(encoding="utf-8"))
        replacement_counts = {
            item["doc_id"]: int(item.get("markdown_replacement_character_count", 0))
            for item in quality_report.get("details", {}).get("documents", [])
        }
        pdf_text_layer_audits = {
            item["doc_id"]: item.get("pdf_text_layer_comparison")
            for item in quality_report.get("details", {}).get("documents", [])
        }
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        replacement_counts = {}

    candidates: list[dict[str, Any]] = []
    for record in registry.get("records", {}).values():
        if record.get("processing", {}).get("status") != "done":
            continue
        payload = _read_structured(record)
        markdown_path = MARKDOWN_DIR / f"{record['doc_id']}.md"
        replacement_count = replacement_counts.get(record["doc_id"])
        if replacement_count is None and markdown_path.is_file():
            try:
                replacement_count = markdown_path.read_text(encoding="utf-8").count("\ufffd")
            except (OSError, UnicodeError):
                replacement_count = 0
        history_errors = [
            str(item.get("details", ""))
            for item in record.get("processing", {}).get("history", [])
            if item.get("status") == "error"
        ]
        candidates.append({
            "record": record,
            "payload": payload,
            "suggested_classification": _classification(record, payload),
            "markdown_path": markdown_path,
            "markdown_size": markdown_path.stat().st_size if markdown_path.is_file() else 0,
            "markdown_replacement_character_count": replacement_count or 0,
            "pdf_text_layer_comparison": pdf_text_layer_audits.get(record["doc_id"]),
            "history_errors": history_errors,
        })

    candidates.sort(key=lambda item: item["record"].get("filename", "").casefold())
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    def add_first(
        predicate: Callable[[dict[str, Any]], bool],
        reason: str,
        *,
        prefer_largest: bool = False,
        prefer_replacements: bool = False,
    ) -> None:
        if len(selected) >= limit:
            return
        matches = [
            item for item in candidates
            if item["record"]["doc_id"] not in used_ids and predicate(item)
        ]
        if not matches:
            return
        if prefer_replacements:
            chosen = max(matches, key=lambda item: item["markdown_replacement_character_count"])
        elif prefer_largest:
            chosen = max(matches, key=lambda item: item["markdown_size"])
        else:
            chosen = matches[0]
        chosen["selection_reason"] = reason
        selected.append(chosen)
        used_ids.add(chosen["record"]["doc_id"])

    add_first(
        lambda item: item["record"].get("reviewed_classification") == "jurisprudence"
        and "1000 CRITERIOS" in item["record"].get("filename", "").upper(),
        "Human-confirmed jurisprudence; includes criterion markers and contents pages.",
    )
    add_first(
        lambda item: item["record"].get("reviewed_classification") == "jurisprudence"
        and "CODIGO PENAL-TOMO I" in item["record"].get("filename", "").upper(),
        "Human-confirmed correction from a filename suggesting legislation to jurisprudence.",
    )
    add_first(
        lambda item: item["record"].get("reviewed_classification") == "legal_doctrine",
        "Human-confirmed legal doctrine; contains quoted provisions and commentary.",
    )
    add_first(
        lambda item: "CONSTITUCION-POLITICA" in item["record"].get("filename", "").upper(),
        "Constitutional normative source with an unreviewed classification.",
    )
    add_first(
        lambda item: item["suggested_classification"] == "legislation"
        and "CODIGO PENAL-COMPRIMIDO" in item["record"].get("filename", "").upper(),
        "Unreviewed legislation proposal for article-boundary inspection.",
    )
    add_first(
        lambda item: item["suggested_classification"] == "jurisprudence"
        and "CASACIONES Y ACUERDOS" in item["record"].get("filename", "").upper(),
        "Unreviewed jurisprudence proposal for decision-structure inspection.",
    )
    add_first(
        lambda item: item["suggested_classification"] == "legal_doctrine"
        and item["record"].get("ocr_assessment", {}).get("required") is True
        and bool(item["history_errors"]),
        "Doctrine/OCR sample with a prior extraction error in processing history.",
    )
    add_first(
        lambda item: item["record"].get("extension", "").lower() == ".docx",
        "DOCX extraction and page-provenance sample.",
    )
    add_first(
        lambda item: item["record"].get("ocr_assessment", {}).get("required") is True,
        "OCR-required long document.",
        prefer_largest=True,
    )
    add_first(
        lambda item: item["markdown_replacement_character_count"] > 0,
        "Markdown contains replacement characters requiring source-level fidelity review.",
        prefer_replacements=True,
    )

    # Fill any unavailable strata without inferring labels or modifying the registry.
    for item in candidates:
        if len(selected) >= limit:
            break
        doc_id = item["record"]["doc_id"]
        if doc_id not in used_ids:
            item["selection_reason"] = "Fallback sample; review classification and source traceability."
            selected.append(item)
            used_ids.add(doc_id)

    return selected


def _chunk_score(chunk: dict[str, Any], sample: dict[str, Any]) -> tuple[int, int]:
    text = str(chunk.get("text", ""))
    record = sample["record"]
    filename = record.get("filename", "").upper()
    category = sample["suggested_classification"]
    score = 0

    if category == "jurisprudence" and chunk.get("structure_type") in {"case_section", "decision"}:
        score += 500
    if category == "legal_doctrine" and chunk.get("structure_type") in {"commentary", "quoted_provision"}:
        score += 500
    if category == "legislation" and chunk.get("structure_type") == "article":
        score += 500
    if "CODIGO PENAL-COMPRIMIDO" in filename and re.search(r"\bART[IÍI]CULO\b", text, re.I):
        score += 300
    if "CONSTITUCI" in filename and chunk.get("structure_type") in {"title", "chapter", "section"}:
        score += 200
    if sample.get("markdown_replacement_character_count"):
        score += 1000 * text.count("\ufffd")
    if chunk.get("source_pages"):
        score += 10

    # Prefer readable passages while retaining a fixed upper bound.
    return score, min(len(text), 1400)


def _select_chunks(samples: list[dict[str, Any]]) -> None:
    by_id = {item["record"]["doc_id"]: item for item in samples}
    selected_ids = set(by_id)
    with (CANDIDATE_INDEX_DIR / "chunks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            chunk = json.loads(line)
            sample = by_id.get(chunk.get("doc_id"))
            if sample is None:
                continue
            if sample.get("chunk") is None or _chunk_score(chunk, sample) > _chunk_score(sample["chunk"], sample):
                sample["chunk"] = chunk

    for doc_id in selected_ids:
        if by_id[doc_id].get("chunk") is None:
            by_id[doc_id]["chunk"] = None


def _structured_units(sample: dict[str, Any]) -> list[dict[str, Any]]:
    units = sample["payload"].get("units", [])
    return units if isinstance(units, list) else []


def _associated_unit_text(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    chunk = sample.get("chunk") or {}
    refs = chunk.get("source_unit_ids", [])
    structured_units = _structured_units(sample)
    matched: list[dict[str, Any]] = []
    for ref in refs:
        suffix = str(ref).rsplit("page-", 1)[-1]
        try:
            ordinal = int(suffix)
        except ValueError:
            continue
        pages = set(chunk.get("source_pages") or [])
        page = ordinal if ordinal in pages else None
        unit = (
            next(
                (item for item in structured_units if item.get("page_number") == page),
                None,
            )
            if page is not None
            else None
        )
        if unit is None and 0 < ordinal <= len(structured_units):
            unit = structured_units[ordinal - 1]
        if unit and unit not in matched:
            matched.append(unit)

    source_text = " ".join(str(unit.get("text", "")) for unit in matched)
    chunk_text = str(chunk.get("text", ""))
    compact = lambda value: re.sub(r"\s+", " ", value).strip()
    trace_match = bool(source_text) and compact(chunk_text) in compact(source_text)
    return matched, trace_match


def _markdown_excerpt(path: Path, page_numbers: list[int]) -> str:
    if not path.is_file():
        return "[Markdown source file missing]"
    text = path.read_text(encoding="utf-8")
    if page_numbers:
        sections = re.split(r"(?=^##\s+Page\s+\d+\s*$)", text, flags=re.MULTILINE | re.IGNORECASE)
        for section in sections:
            match = PAGE_HEADING_RE.search(section)
            if match and int(match.group(1)) in page_numbers:
                body = section[match.end():].strip()
                return body[:1200]
    match = re.search(r"^##\s+Extracted text\s*$", text, re.MULTILINE | re.IGNORECASE)
    if match:
        return text[match.end():].strip()[:1200]
    return text[:1200]


def _clip(text: str, limit: int = 1200) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit] + " … [truncated for review packet]"


def build_review_packet(
    limit: int = 10,
    output_dir: Path = RETROSPECTIVE_PILOT_DIR,
) -> dict[str, Any]:
    """Generate a reproducible, source-anchored review packet without labeling ground truth."""
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    if not (CANDIDATE_INDEX_DIR / "chunks.jsonl").is_file():
        raise FileNotFoundError(
            f"Candidate chunks are missing: {CANDIDATE_INDEX_DIR / 'chunks.jsonl'}"
        )

    samples = _choose_records(limit)
    _select_chunks(samples)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_fields = [
        "document_id", "filename", "source_format", "ocr_required",
        "markdown_replacement_character_count",
        "pdf_text_layer_comparison_status",
        "structured_suggested_classification", "existing_human_classification",
        "classification_accepted", "corrected_classification",
        "source_text_fidelity", "structure_label_correct", "hierarchy_correct",
        "page_reference_correct", "review_notes",
    ]
    rows: list[dict[str, Any]] = []
    packet: list[str] = [
        "# Paquete de revisión retrospectiva",
        "",
        f"Generado: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Este paquete presenta evidencia de los artefactos existentes para una revisión humana.",
        "No crea etiquetas de referencia ni modifica el registro. Las decisiones nuevas quedan vacías en `review-labels.csv`.",
        "La sugerencia estructurada y el tipo del chunk son resultados del sistema, no verdad de terreno.",
        "",
        f"Muestras incluidas: {len(samples)} (límite solicitado: {limit}).",
        "",
        "Para cada documento revise: fidelidad de texto, clase documental, límites del fragmento, jerarquía y página de origen.",
        "Los textos largos se recortan únicamente en este paquete; los archivos de origen y los índices no se alteran.",
        "",
    ]

    for position, sample in enumerate(samples, start=1):
        record = sample["record"]
        chunk = sample.get("chunk") or {}
        payload = sample["payload"]
        reviewed = record.get("reviewed_classification")
        pages = chunk.get("source_pages") or []
        units, trace_match = _associated_unit_text(sample)
        structured_excerpt = "\n\n".join(str(unit.get("text", "")) for unit in units)
        markdown_excerpt = _markdown_excerpt(sample["markdown_path"], pages)
        chunk_text = str(chunk.get("text", ""))
        metadata = payload.get("extraction", {})
        errors = sample["history_errors"]

        packet.extend([
            f"## Muestra {position}: {record.get('filename', 'unknown filename')}",
            "",
            f"- Document ID: `{record['doc_id']}`",
            f"- SHA-256: `{record.get('sha256', payload.get('source', {}).get('sha256', 'unknown'))}`",
            f"- Formato: `{record.get('extension', 'unknown')}`; OCR requerido: `{record.get('ocr_assessment', {}).get('required', 'unknown')}`",
            f"- Propuesta estructurada: `{sample['suggested_classification']}`",
            f"- Etiqueta humana ya confirmada: `{reviewed or 'ninguna'}`",
            f"- Motivo de inclusión: {sample['selection_reason']}",
            f"- Historial de errores previo: `{len(errors)}`; método de extracción actual: `{metadata.get('method', metadata.get('extraction_method', 'not recorded'))}`",
            f"- Caracteres de reemplazo U+FFFD en Markdown: `{sample['markdown_replacement_character_count']}`",
            f"- Comparación con la capa de texto PDF: `{(sample.get('pdf_text_layer_comparison') or {}).get('status', 'not assessed')}`",
            f"- Chunk candidato: `{chunk.get('chunk_id', 'not found')}`; tipo: `{chunk.get('structure_type', 'not found')}`; referencia: `{chunk.get('structure_reference') or 'none'}`",
            f"- Páginas del chunk: `{pages or 'not recorded'}`; unidades de origen: `{chunk.get('source_unit_ids', [])}`",
            f"- Ruta jerárquica del chunk: `{chunk.get('hierarchy_path', [])}`",
            f"- Coincidencia del chunk dentro de las unidades estructuradas asociadas: `{'yes' if trace_match else 'no / unavailable'}`",
            "",
            "### Extracto de Markdown (página fuente)",
            "",
            "```text",
            _clip(markdown_excerpt),
            "```",
            "",
            "### Extracto estructurado (unidad fuente)",
            "",
            "```text",
            _clip(structured_excerpt or "[No associated structured unit found]"),
            "```",
            "",
            "### Chunk candidato",
            "",
            "```text",
            _clip(chunk_text or "[No candidate chunk found]"),
            "```",
            "",
            "### Revisión humana",
            "",
            "- ¿La clasificación sugerida corresponde al tipo real del documento?",
            "- ¿El chunk representa una unidad estructural correcta y no una cita o entrada de índice?",
            "- ¿La jerarquía y la página de origen son correctas?",
            "- Observaciones: ______________________________",
            "",
        ])

        rows.append({
            "document_id": record["doc_id"],
            "filename": record.get("filename", ""),
            "source_format": record.get("extension", ""),
            "ocr_required": record.get("ocr_assessment", {}).get("required", ""),
            "markdown_replacement_character_count": sample["markdown_replacement_character_count"],
            "pdf_text_layer_comparison_status": (
                (sample.get("pdf_text_layer_comparison") or {}).get("status", "")
            ),
            "structured_suggested_classification": sample["suggested_classification"],
            "existing_human_classification": reviewed or "",
            "classification_accepted": "",
            "corrected_classification": "",
            "source_text_fidelity": "",
            "structure_label_correct": "",
            "hierarchy_correct": "",
            "page_reference_correct": "",
            "review_notes": "",
        })

    packet_path = output_dir / "review-packet.md"
    packet_path.write_text("\n".join(packet), encoding="utf-8")
    labels_path = output_dir / "review-labels.csv"
    with labels_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=label_fields)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "source_index": str(CANDIDATE_INDEX_DIR),
        "ground_truth_created": False,
        "documents": [
            {
                "doc_id": item["record"]["doc_id"],
                "filename": item["record"].get("filename"),
                "selection_reason": item["selection_reason"],
                "candidate_chunk_id": (item.get("chunk") or {}).get("chunk_id"),
            }
            for item in samples
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "sample_count": len(samples),
        "review_packet": str(packet_path),
        "labels_csv": str(labels_path),
        "manifest": str(output_dir / "manifest.json"),
    }
