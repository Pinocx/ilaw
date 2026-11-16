"""Create source-linked metadata candidates without changing legal records."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REGISTRY_FILE, REPORTS_DIR, STRUCTURED_DIR
from .registry import load_registry


TITLE_NOISE_RE = re.compile(
    r"^(?:\d+|[\W_]*|(?:[ÍI]NDICE|CONTENIDO|INTRODUCCI[ÓO]N|"
    r"PRESENTACI[ÓO]N|PR[ÓO]LOGO|ANTECEDENTES|EDICI[ÓO]N(?:\s+OFICIAL)?))$",
    re.IGNORECASE,
)
DOCUMENT_IDENTIFIER_RE = re.compile(
    r"\b(?:Ley|Decreto\s+(?:Legislativo|Supremo|de\s+Urgencia)|"
    r"Resoluci[oó]n\s+(?:Ministerial|Suprema|Directoral|Administrativa|"
    r"del\s+Tribunal\s+Constitucional|Legislativa))\s+"
    r"(?:N[.°ºo]\s*)?\d+(?:[-/][A-Z0-9]+)*",
    re.IGNORECASE,
)
DATE_MENTION_RE = re.compile(
    r"\b(?:\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|"
    r"julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
    r"\s+de\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4}|"
    r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    r"septiembre|setiembre|octubre|noviembre|diciembre)\s+de?\s*\d{4})\b",
    re.IGNORECASE,
)
AUTHORITY_PATTERNS = (
    "Congreso de la República",
    "Tribunal Constitucional",
    "Poder Judicial",
    "Ministerio Público",
    "Ministerio de Justicia y Derechos Humanos",
    "Presidencia de la República",
    "Defensoría del Pueblo",
    "Jurado Nacional de Elecciones",
    "Contraloría General de la República",
)
MAX_CHARS_PER_UNIT = 12000


def _evidence(unit: dict[str, Any], value: str, method: str) -> dict[str, Any]:
    text = str(unit.get("text") or "")
    position = text.casefold().find(value.casefold())
    start = max(0, position - 120) if position >= 0 else 0
    end = min(len(text), (position + len(value) + 120) if position >= 0 else 240)
    return {
        "value": value,
        "source_unit_id": unit.get("unit_id"),
        "source_page": unit.get("page_number"),
        "evidence_excerpt": text[start:end].replace("\n", " ").strip(),
        "proposal_method": method,
        "review_status": "suggested_requires_review",
    }


def _collect_candidates(units: list[dict[str, Any]], max_units: int) -> dict[str, list[dict[str, Any]]]:
    title_candidates: list[dict[str, Any]] = []
    identifier_candidates: list[dict[str, Any]] = []
    date_mentions: list[dict[str, Any]] = []
    authority_mentions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for unit in units[:max_units]:
        text = str(unit.get("text") or "")[:MAX_CHARS_PER_UNIT]
        for line_number, raw_line in enumerate(text.splitlines()):
            if re.search(r"(?:\t+|\.{3,}|…{2,}|_{3,})\s*\d+\s*$", raw_line):
                continue
            line = re.sub(r"\s+", " ", raw_line).strip(" \t-–—")
            letters = [character for character in line if character.isalpha()]
            uppercase_ratio = (
                sum(character.isupper() for character in letters) / len(letters)
                if letters else 0.0
            )
            if (
                line_number < 12
                and 5 <= len(line) <= 160
                and uppercase_ratio >= 0.65
                and not TITLE_NOISE_RE.fullmatch(line)
                and sum(character.isalpha() for character in line) >= 4
            ):
                key = ("title", line.casefold())
                if key not in seen:
                    title_candidates.append(_evidence(unit, line, "first_units_title_line"))
                    seen.add(key)

        for match in DOCUMENT_IDENTIFIER_RE.finditer(text):
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            key = ("identifier", value.casefold())
            if key not in seen:
                identifier_candidates.append(_evidence(unit, value, "legal_identifier_pattern_v1"))
                seen.add(key)

        for match in DATE_MENTION_RE.finditer(text):
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            key = ("date", value.casefold())
            if key not in seen:
                date_mentions.append(_evidence(unit, value, "date_mention_pattern_v1"))
                seen.add(key)

        for authority in AUTHORITY_PATTERNS:
            for match in re.finditer(re.escape(authority), text, re.IGNORECASE):
                value = match.group(0)
                key = ("authority", value.casefold())
                if key not in seen:
                    authority_mentions.append(_evidence(unit, value, "authority_lexicon_v1"))
                    seen.add(key)

    return {
        "title_candidates": title_candidates[:20],
        "document_identifier_candidates": identifier_candidates,
        "date_mentions": date_mentions,
        "issuing_authority_mentions": authority_mentions,
    }


def build_metadata_proposals(
    *,
    registry_path: Path = REGISTRY_FILE,
    structured_dir: Path = STRUCTURED_DIR,
    output_path: Path = REPORTS_DIR / "metadata-proposals.json",
    max_units: int = 3,
) -> dict[str, Any]:
    """Propose source-grounded metadata; never write proposals into the registry."""
    if max_units < 1:
        raise ValueError("max_units must be at least 1.")

    registry = load_registry(registry_path)
    records = list(registry.get("records", {}).values())
    documents: list[dict[str, Any]] = []
    candidate_counts: Counter[str] = Counter()
    missing_artifacts = 0

    for record in records:
        if record.get("processing", {}).get("status") != "done":
            continue
        doc_id = record.get("doc_id")
        structured_path = Path(structured_dir) / f"{doc_id}.json"
        try:
            structured = json.loads(structured_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            missing_artifacts += 1
            documents.append({
                "doc_id": doc_id,
                "filename": record.get("filename"),
                "status": "structured_source_unavailable",
            })
            continue

        units = structured.get("units", [])
        proposals = _collect_candidates(units if isinstance(units, list) else [], max_units)
        for field, values in proposals.items():
            if values:
                candidate_counts[field] += 1
        documents.append({
            "doc_id": doc_id,
            "filename": record.get("filename"),
            "source_units_examined": min(len(units), max_units) if isinstance(units, list) else 0,
            "classification_status": (
                "reviewed" if record.get("reviewed_classification") else "pending_review"
            ),
            "proposals": proposals,
            "fields_not_inferred": [
                "jurisdiction",
                "publication_date",
                "effective_date",
                "legal_status",
            ],
        })

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Deterministic metadata candidates from bounded first source units; not authoritative metadata.",
        "source_units_per_document": max_units,
        "maximum_characters_per_unit": MAX_CHARS_PER_UNIT,
        "documents_examined": len(documents),
        "documents_missing_structured_source": missing_artifacts,
        "documents_with_candidates_by_field": dict(candidate_counts),
        "registry_mutated": False,
        "source_documents_mutated": False,
        "documents": documents,
        "output_json": str(Path(output_path).resolve()),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
