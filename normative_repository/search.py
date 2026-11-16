"""Search candidate chunks through FTS5 or a source-JSONL fallback."""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any

from .config import CANDIDATE_INDEX_DIR
from .text_index import (
    TOKEN_RE,
    _fts_query,
    can_use_search_index,
    iter_indexed_chunks,
    normalize_search_text,
    search_index_status,
)


_normalize = normalize_search_text


def _snippet(text: str, query: str, radius: int = 180) -> str:
    normalized_text = _normalize(text)
    normalized_query = _normalize(query.strip())
    position = normalized_text.find(normalized_query)
    if position < 0:
        terms = [term for term in TOKEN_RE.findall(normalized_query) if term]
        positions = [normalized_text.find(term) for term in terms]
        positions = [position for position in positions if position >= 0]
        position = min(positions, default=0)
    start = max(0, position - radius)
    end = min(len(text), position + len(query) + radius)
    prefix = "…" if start else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end].replace("\n", " ").strip() + suffix


def _load_documents(path: Path) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = record.get("doc_id")
                if doc_id:
                    documents[doc_id] = record
    except FileNotFoundError:
        return {}
    return documents


def search_chunks(
    query: str,
    *,
    index_dir: Path = CANDIDATE_INDEX_DIR,
    limit: int = 10,
    classification: str | None = None,
    structure_type: str | None = None,
    exact_phrase: bool = False,
    reviewed_only: bool = False,
) -> dict[str, Any]:
    """Search chunks with deterministic matching and return source-traceable hits.

    Queries use AND semantics across normalized terms. Exact phrase mode requires
    the complete normalized query to occur in the searchable text or metadata.
    """
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")
    if limit < 1 or limit > 1000:
        raise ValueError("Search limit must be between 1 and 1000.")

    chunks_path = index_dir / "chunks.jsonl"
    documents = _load_documents(index_dir / "documents.jsonl")
    terms = list(dict.fromkeys(_normalize(term) for term in TOKEN_RE.findall(query)))
    if not terms:
        raise ValueError("Search query must contain at least one letter or number.")
    normalized_query = _normalize(query)
    top_hits: list[tuple[int, int, dict[str, Any]]] = []
    total_matches = 0
    malformed_rows = 0
    chunks_scanned = 0
    index_status = search_index_status(index_dir)
    use_search_index = (
        index_status["available"]
        and can_use_search_index(terms, normalized_query, exact_phrase)
    )

    if use_search_index:
        chunk_rows = iter_indexed_chunks(
            index_dir,
            _fts_query(terms, normalized_query, exact_phrase),
        )
        search_backend = "sqlite_fts5_trigram"
        malformed_rows = int(index_status.get("malformed_rows", 0))
    else:
        try:
            stream = chunks_path.open("r", encoding="utf-8")
        except OSError as exc:
            raise FileNotFoundError(f"Cannot open candidate chunks: {chunks_path}") from exc

        def read_jsonl_chunks():
            with stream:
                for row_number, line in enumerate(stream):
                    if not line.strip():
                        continue
                    try:
                        yield row_number, json.loads(line)
                    except json.JSONDecodeError:
                        nonlocal_malformed[0] += 1

        nonlocal_malformed = [0]
        chunk_rows = read_jsonl_chunks()
        search_backend = "jsonl_scan"

    for row_number, chunk in chunk_rows:
        chunks_scanned += 1
        doc = documents.get(chunk.get("doc_id"), {})
        reviewed_classification = chunk.get("reviewed_classification")
        suggested_classification = chunk.get("suggested_classification")
        effective_classification = reviewed_classification or suggested_classification
        classification_status = chunk.get("classification_status", "pending_review")

        if not use_search_index:
            malformed_rows = nonlocal_malformed[0]
        if reviewed_only and classification_status != "reviewed":
            continue
        if classification and effective_classification != classification:
            continue
        if structure_type and chunk.get("structure_type") != structure_type:
            continue

        text = str(chunk.get("text", ""))
        structure_values = [
            chunk.get("structure_reference") or "",
            chunk.get("parent_reference") or "",
            " ".join(chunk.get("hierarchy_path") or []),
            doc.get("filename") or "",
        ]
        normalized_text = _normalize(text)
        normalized_structure = _normalize(" ".join(structure_values[:3]))
        normalized_filename = _normalize(str(structure_values[3]))
        normalized_metadata = f"{normalized_structure} {normalized_filename}"
        searchable = f"{normalized_text} {normalized_metadata}"
        if exact_phrase:
            if (
                normalized_query not in normalized_text
                and normalized_query not in normalized_metadata
            ):
                continue
            matched_terms = len(terms)
        else:
            matched_terms = sum(term in searchable for term in terms)
            if matched_terms != len(terms):
                continue

        text_phrase_match = normalized_query in normalized_text
        metadata_phrase_match = normalized_query in normalized_metadata
        metadata_match = metadata_phrase_match
        score = (
            (100 if text_phrase_match else 0)
            + (30 if metadata_match else 0)
            + sum(
                3 * normalized_text.count(term)
                + normalized_metadata.count(term)
                for term in terms
            )
        )
        matched_fields = []

        def field_matches(value: str) -> bool:
            if exact_phrase:
                return normalized_query in value
            return all(term in value for term in terms)

        if field_matches(normalized_text):
            matched_fields.append("text")
        if field_matches(normalized_structure):
            matched_fields.append("structure")
        if field_matches(normalized_filename):
            matched_fields.append("document_title")
        source_pages = chunk.get("source_pages") or []
        hit = {
            "score": score,
            "doc_id": chunk.get("doc_id"),
            "filename": doc.get("filename"),
            "chunk_id": chunk.get("chunk_id"),
            "structure_type": chunk.get("structure_type"),
            "structure_reference": chunk.get("structure_reference"),
            "parent_reference": chunk.get("parent_reference"),
            "hierarchy_path": chunk.get("hierarchy_path", []),
            "source_pages": source_pages,
            "source_unit_ids": chunk.get("source_unit_ids", []),
            "suggested_classification": suggested_classification,
            "reviewed_classification": reviewed_classification,
            "classification_status": classification_status,
            "legal_verification_status": chunk.get(
                "legal_verification_status", "not_verified"
            ),
            "matched_terms": matched_terms,
            "matched_fields": matched_fields,
            "text": text,
            "snippet": _snippet(text, query),
        }
        total_matches += 1
        entry = (score, -row_number, hit)
        if len(top_hits) < limit:
            heapq.heappush(top_hits, entry)
        elif entry[:2] > top_hits[0][:2]:
            heapq.heapreplace(top_hits, entry)

    if not use_search_index:
        malformed_rows = nonlocal_malformed[0]

    top_hits.sort(key=lambda entry: (-entry[0], -entry[1]))
    hits = [entry[2] for entry in top_hits]
    return {
        "query": query,
        "query_mode": "exact_phrase" if exact_phrase else "all_terms",
        "index_dir": str(index_dir),
        "chunks_scanned": chunks_scanned,
        "search_backend": search_backend,
        "search_index_fallback_reason": (
            None if use_search_index else index_status.get("reason")
            or "query_contains_terms_shorter_than_three_characters"
        ),
        "malformed_rows": malformed_rows,
        "total_matches": total_matches,
        "returned_matches": len(hits),
        "filters": {
            "classification": classification,
            "structure_type": structure_type,
            "reviewed_only": reviewed_only,
        },
        "results": hits[:limit],
        "warning": (
            "Scores are heuristic, not confidence values. Candidate index results "
            "are derived data; classification and legal verification fields must "
            "be checked before legal use."
        ),
    }
