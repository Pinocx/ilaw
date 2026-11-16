
"""Adaptive legal document segmentation."""

from __future__ import annotations

import re
from typing import Any


DEFAULT_MAX_CHARS = 1800


# Legal article markers.
ARTICLE_RE = re.compile(
    r"^\s*(?:ART[IÍI]CULO|ARTICULO|ART\.?)\s*"
    r"(?P<number>ÚNICO|UNICO|\d+(?:[.-]\d+)*(?:[A-Z])?)"
    r"\s*(?:[º°o.]|[-:])?\s*(?P<title>.*)$",
    re.IGNORECASE,
)

TOC_LINE_RE = re.compile(
    r"^\s*(?P<label>.+?)\s*(?:\.{3,}|…{2,}|_{3,}|\t+)\s*\d+\s*$"
)
JURISPRUDENCE_ITEM_RE = re.compile(r"^\s*J\s*\.?\s*(?P<number>\d{1,4})\s*$", re.I)
QUOTED_PROVISION_RE = re.compile(
    r"^\s*[^\w]{0,6}(?:ART[IÍI]CULO|ARTICULO)\s*"
    r"(?P<number>ÚNICO|UNICO|\d+(?:[.-]\d+)*(?:[A-Z])?)"
    r"(?:\s*[º°o.]?)\s*[^\w]*$",
    re.I,
)
COMMENTARY_HEADING_RE = re.compile(r"\bCOMENTARIOS?\s*:\s*$", re.I)


# Hierarchical headings and their levels.
HIERARCHY_PATTERNS = [
    (
        1,
        "book",
        re.compile(r"^\s*(?:LIBRO|TOMO)\s+(?:[IVXLCDM]+|\d+)\b.*$", re.I),
    ),
    (
        2,
        "title",
        re.compile(r"^\s*T[IÍ]TULO\s+(?:[IVXLCDM]+|\d+)\b.*$", re.I),
    ),
    (
        3,
        "chapter",
        re.compile(r"^\s*CAP[IÍ]TULO\s+(?:[IVXLCDM]+|\d+)\b.*$", re.I),
    ),
    (
        4,
        "section",
        re.compile(r"^\s*(?:SECCI[ÓO]N|SUBCAP[IÍ]TULO)\s+.*$", re.I),
    ),
]

DISPOSITION_RE = re.compile(
    r"^\s*DISPOSICIONES\s+"
    r"(?:COMPLEMENTARIAS|TRANSITORIAS|FINALES|DEROGATORIAS|MODIFICATORIAS)"
    r".*$",
    re.IGNORECASE,
)

JURISPRUDENCE_HEADINGS = {
    "VISTOS": "case_section",
    "VISTA": "case_section",
    "CONSIDERANDO": "case_section",
    "FUNDAMENTOS": "case_section",
    "FUNDAMENTOS DE HECHO": "case_section",
    "FUNDAMENTOS DE DERECHO": "case_section",
    "ANTECEDENTES": "case_section",
    "PARTE RESOLUTIVA": "decision",
    "RESUELVE": "decision",
    "FALLO": "decision",
    "DECISIÓN": "decision",
    "DECISION": "decision",
}


def _clean_line(value: str) -> str:
    """Normalize whitespace without changing the original stored text."""
    return re.sub(r"\s+", " ", value).strip()


def _is_hierarchy_heading(line: str) -> bool:
    """Reject citation-like lines that happen to begin with a hierarchy word."""
    if len(line) > 160:
        return False
    citation_markers = re.compile(
        r"\b(?:19|20)\d{2}\b|\bpp?\.?\s*\d|"
        r"\b(?:edici[oó]n|editorial|ISBN|afirma|señala)\b",
        re.I,
    )
    return not citation_markers.search(line)


def _hierarchy_reference(line: str) -> str:
    """Keep structural references concise and stable across page text noise."""
    patterns = (
        r"^\s*(?:LIBRO|TOMO)\s+(?:[IVXLCDM]+|\d+(?:-[A-Z])?)",
        r"^\s*T[IÍ]TULO\s+(?:[IVXLCDM]+|\d+)",
        r"^\s*CAP[IÍ]TULO\s+(?:[IVXLCDM]+|\d+)",
        r"^\s*(?:SECCI[ÓO]N|SUBCAP[IÍ]TULO)\s+[^:.;]{1,80}",
    )
    for pattern in patterns:
        match = re.match(pattern, line, re.I)
        if match:
            prefix = match.group(0).strip()
            suffix = line[match.end():].strip(" :-–—.")
            if suffix and len(suffix) <= 80:
                return f"{prefix} {suffix}"
            return prefix
    return _clean_line(line)

def _is_article_heading(line: str, article: re.Match[str]) -> bool:
    """Validate whether an article marker is a genuine legal heading."""

    stripped = line.lstrip()

    # Require a capitalized structural marker.
    valid_prefixes = (
        "Artículo",
        "ARTÍCULO",
        "ARTICULO",
        "Art.",
        "ART.",
    )
    if not stripped.startswith(valid_prefixes):
        return False

    title = (article.group("title") or "").strip()

    # Ignore OCR ornaments around a standalone article number.
    title = re.sub(r"^[^\w]+|[^\w]+$", "", title).strip()
    if not title:
        return bool(re.fullmatch(
            r"\s*(?:ART[IÍI]CULO|ARTICULO|ART\.?)\s*"
            r"(?:ÚNICO|UNICO|\d+(?:[.-]\d+)*(?:[A-Z])?)"
            r"\s*(?:[º°o.]|[-:])?\s*[^\w]*",
            line,
            re.I,
        ))
    if len(title) < 3:
        return False

    # Citations commonly continue with lowercase phrases such as "del Código".
    if not title[0].isupper():
        return False
    if re.match(
        r"^(?:del?\s+(?:c[oó]digo|presente|citado)|de la\s+(?:ley|norma)|"
        r"previsto en|conforme a)\b",
        title,
        re.I,
    ):
        return False

    # Reject references to subsections and constitutional provisions.
    if re.match(
        r"^(?:inc\.?|inciso|literal|numeral|apartado|párrafo)\b",
        title,
        re.IGNORECASE,
    ):
        return False

    # Reject table-of-contents dotted leaders.
    if re.search(r"(?:\.\s*){4,}|\.{3,}|…{2,}|_{3,}", title):
        return False

    # Reject headings that end with a page number after a dotted leader.
    if re.search(r"(?:\.\s*){3,}\s*\d+\s*$", title):
        return False

    # Reject text that clearly starts a quotation or citation.
    if title.startswith(("'", '"', "“", "‘")):
        return False

    return True


def _detect_marker(
    line: str,
    classification_type: str = "unknown",
) -> dict[str, Any] | None:
    """Identify a recognized structural heading."""

    toc_entry = TOC_LINE_RE.match(line)
    if toc_entry:
        return {
            "kind": "table_of_contents_entry",
            "reference": _clean_line(toc_entry.group("label")),
            "level": 6,
        }

    if classification_type == "jurisprudence":
        case_item = JURISPRUDENCE_ITEM_RE.match(line)
        if case_item:
            return {
                "kind": "case_section",
                "reference": f"Criterio J {case_item.group('number')}",
                "level": 4,
            }

    if classification_type == "legal_doctrine":
        if COMMENTARY_HEADING_RE.search(line):
            return {
                "kind": "commentary",
                "reference": "Commentary",
                "level": 5,
            }
        quoted_provision = QUOTED_PROVISION_RE.match(line)
        if quoted_provision:
            return {
                "kind": "quoted_provision",
                "reference": f"Artículo {quoted_provision.group('number')}",
                "level": 5,
            }

    if classification_type == "legislation":
        article = ARTICLE_RE.match(line)
        if article and _is_article_heading(line, article):
            number = article.group("number").strip()
            return {
                "kind": "article",
                "reference": f"Artículo {number}",
                "level": 5,
            }

    for level, kind, pattern in HIERARCHY_PATTERNS:
        if pattern.match(line) and _is_hierarchy_heading(line):
            return {
                "kind": kind,
                "reference": _hierarchy_reference(line),
                "level": level,
            }

    if DISPOSITION_RE.match(line):
        return {
            "kind": "disposition",
            "reference": _clean_line(line),
            "level": 4,
        }

    normalized = _clean_line(line).rstrip(":").upper()
    if normalized in JURISPRUDENCE_HEADINGS:
        return {
            "kind": JURISPRUDENCE_HEADINGS[normalized],
            "reference": _clean_line(line),
            "level": 5,
        }

    return None


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split oversized text at sentence or word boundaries."""

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?;])\s+", text)
    result: list[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if current:
                result.append(current.strip())
                current = ""

            words = sentence.split()
            part = ""

            for word in words:
                candidate = f"{part} {word}".strip()
                if len(candidate) > max_chars and part:
                    result.append(part.strip())
                    part = word
                else:
                    part = candidate

            if part:
                current = part
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) > max_chars and current:
            result.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current:
        result.append(current.strip())

    return result


def segment_document(
    doc_id: str,
    units: list[dict[str, Any]],
    classification: dict[str, Any] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[dict[str, Any]]:
    """
    Segment extracted document units into searchable chunks.

    The input units are preserved. This function only creates a
    separate representation for indexing.
    """

    if max_chars < 200:
        raise ValueError("max_chars must be at least 200.")
    if isinstance(classification, dict):
        classification_type = (
            classification.get("suggested")
            or classification.get("suggested_classification")
            or classification.get("classification")
            or "unknown"
        )
    elif isinstance(classification, str):
        classification_type = classification
    else:
        classification_type = "unknown"

    blocks: list[dict[str, Any]] = []
    hierarchy: list[tuple[int, str]] = []

    active_kind = "paragraph"
    active_reference: str | None = None
    active_parent_reference: str | None = None
    active_hierarchy_path: list[str] = []

    buffer: list[str] = []
    buffer_pages: set[int] = set()
    buffer_unit_ids: set[str] = set()

    def flush_buffer() -> None:
        nonlocal buffer, buffer_pages, buffer_unit_ids

        text = _clean_line(" ".join(buffer))

        if text:
            blocks.append({
                "text": text,
                "structure_type": active_kind,
                "structure_reference": active_reference,
                "parent_reference": active_parent_reference,
                "hierarchy_path": list(active_hierarchy_path),
                "pages": sorted(buffer_pages),
                "source_unit_ids": sorted(buffer_unit_ids),
            })

        buffer = []
        buffer_pages = set()
        buffer_unit_ids = set()

    for unit in units:
        unit_text = unit.get("text", "")

        if not isinstance(unit_text, str) or not unit_text.strip():
            continue

        unit_id = str(unit.get("unit_id", ""))
        page_number = unit.get("page_number")

        for raw_line in unit_text.replace("\f", "\n").splitlines():
            line = _clean_line(raw_line)

            if not line:
                flush_buffer()
                continue

            # Keep the raw line for TOC detection: tab separators are meaningful
            # in DOCX-generated tables of contents but are normalized by _clean_line.
            marker = _detect_marker(
                raw_line,
                classification_type=classification_type,
            )
            if marker:
                flush_buffer()

                level = marker["level"]
                hierarchy = [
                    item for item in hierarchy if item[0] < level
                ]

                # Capture the parent before adding the current heading.
                active_parent_reference = (
                    " / ".join(item[1] for item in hierarchy) or None
                )

                if marker["kind"] in {
                    "book", "title", "chapter", "section", "disposition"
                } or (
                    marker["kind"] == "case_section"
                    and marker["reference"].startswith("Criterio J ")
                ):
                    hierarchy.append((level, marker["reference"]))

                active_hierarchy_path = [item[1] for item in hierarchy]
                if marker["kind"] not in {
                    "book", "title", "chapter", "section", "disposition"
                } and not (
                    marker["kind"] == "case_section"
                    and marker["reference"].startswith("Criterio J ")
                ):
                    active_hierarchy_path.append(marker["reference"])

                active_kind = marker["kind"]
                active_reference = marker["reference"]
            if line:
                buffer.append(line)

                if page_number is not None:
                    buffer_pages.add(page_number)

                if unit_id:
                    buffer_unit_ids.add(unit_id)

            # Prevent very large paragraphs from accumulating indefinitely.
            if sum(len(part) for part in buffer) > max_chars * 2:
                flush_buffer()

    flush_buffer()

    # Merge consecutive blocks belonging to the same structural unit.
    merged: list[dict[str, Any]] = []

    for block in blocks:
        if merged:
            previous = merged[-1]
            same_structure = (
                previous["structure_type"] == block["structure_type"]
                and previous["structure_reference"] == block["structure_reference"]
                and previous["parent_reference"] == block["parent_reference"]
            )

            combined_length = len(previous["text"]) + len(block["text"]) + 1

            if same_structure and combined_length <= max_chars:
                previous["text"] += "\n\n" + block["text"]
                previous["pages"] = sorted(
                    set(previous["pages"]) | set(block["pages"])
                )
                previous["source_unit_ids"] = sorted(
                    set(previous["source_unit_ids"])
                    | set(block["source_unit_ids"])
                )
                continue

        merged.append(block.copy())

    # Split oversized blocks and create stable, ordered chunk identifiers.
    chunks: list[dict[str, Any]] = []

    for block in merged:
        parts = _split_long_text(block["text"], max_chars)

        for part_number, part in enumerate(parts, start=1):
            index = len(chunks) + 1

            if block["structure_type"] == "article":
                confidence = 0.95
            elif block["structure_type"] in {
                "quoted_provision", "table_of_contents_entry", "case_section"
            }:
                confidence = 0.85
            elif block["structure_type"] == "paragraph":
                confidence = 0.55
            else:
                confidence = 0.80

            chunks.append({
                "chunk_id": f"{doc_id}:chunk-{index:06d}",
                "doc_id": doc_id,
                "structure_type": block["structure_type"],
                "structure_reference": block["structure_reference"],
                "parent_reference": block["parent_reference"],
                "hierarchy_path": block["hierarchy_path"],
                "part_number": part_number,
                "part_count": len(parts),
                "start_page": min(block["pages"]) if block["pages"] else None,
                "end_page": max(block["pages"]) if block["pages"] else None,
                "source_pages": block["pages"],
                "source_unit_ids": block["source_unit_ids"],
                "text": part,
                "chunking_method": "adaptive_rules_v1",
                "segmentation_confidence": confidence,
                "suggested_classification": (
                    classification.get("suggested")
                    if isinstance(classification, dict)
                    else classification if isinstance(classification, str)
                    else None
                ),
                "legal_verification_status": "not_verified",
            })

    return chunks
