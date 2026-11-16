"""Build and query a persistent local FTS5 index for candidate chunks."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Iterator


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
SEARCH_DATABASE_NAME = "search.sqlite3"
SEARCH_SCHEMA_VERSION = "1"


def normalize_search_text(value: str) -> str:
    """Normalize text for case- and accent-insensitive Spanish matching."""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _read_documents(path: Path) -> dict[str, str]:
    documents: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = row.get("doc_id")
            if doc_id:
                documents[doc_id] = str(row.get("filename") or "")
    return documents


def _source_state(index_dir: Path) -> dict[str, dict[str, int]]:
    state: dict[str, dict[str, int]] = {}
    for filename in ("documents.jsonl", "chunks.jsonl"):
        source = index_dir / filename
        stat = source.stat()
        state[filename] = {
            "size_bytes": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
        }
    return state


def search_index_status(index_dir: Path) -> dict[str, Any]:
    """Return whether the persistent search index matches its JSONL sources."""
    database_path = index_dir / SEARCH_DATABASE_NAME
    if not database_path.is_file():
        return {"available": False, "reason": "search_index_not_built"}

    try:
        current_sources = _source_state(index_dir)
        connection = sqlite3.connect(database_path)
        try:
            rows = dict(connection.execute("SELECT key, value FROM metadata"))
        finally:
            connection.close()
        if rows.get("schema_version") != SEARCH_SCHEMA_VERSION:
            return {"available": False, "reason": "search_index_schema_mismatch"}
        indexed_sources = json.loads(rows["source_state"])
        if indexed_sources != current_sources:
            return {"available": False, "reason": "search_index_stale"}
    except (OSError, sqlite3.Error, KeyError, json.JSONDecodeError):
        return {"available": False, "reason": "search_index_unreadable"}

    return {
        "available": True,
        "reason": None,
        "database_path": str(database_path),
        "malformed_rows": int(rows.get("malformed_rows", "0")),
        "chunks_indexed": int(rows.get("chunks_indexed", "0")),
    }


def _fts_query(terms: list[str], normalized_query: str, exact_phrase: bool) -> str:
    if exact_phrase:
        return '"' + normalized_query.replace('"', '""') + '"'
    return " AND ".join('"' + term.replace('"', '""') + '"' for term in terms)


def can_use_search_index(terms: list[str], normalized_query: str, exact_phrase: bool) -> bool:
    """Trigram search requires at least one query term of three characters."""
    if exact_phrase:
        return len(normalized_query) >= 3
    return bool(terms) and all(len(term) >= 3 for term in terms)


def iter_indexed_chunks(
    index_dir: Path,
    match_query: str,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield candidate chunks from FTS in original source order."""
    database_path = index_dir / SEARCH_DATABASE_NAME
    connection = sqlite3.connect(database_path)
    try:
        cursor = connection.execute(
            """
            SELECT chunks.row_id, chunks.payload
            FROM chunk_fts
            JOIN chunks ON chunks.row_id = chunk_fts.rowid
            WHERE chunk_fts MATCH ?
            ORDER BY chunks.row_id
            """,
            (match_query,),
        )
        for row_id, payload in cursor:
            yield row_id, json.loads(payload)
    finally:
        connection.close()


def build_text_search_index(index_dir: Path) -> dict[str, Any]:
    """Atomically build a trigram FTS5 index from candidate JSONL artifacts."""
    index_dir = Path(index_dir)
    documents_path = index_dir / "documents.jsonl"
    chunks_path = index_dir / "chunks.jsonl"
    if not documents_path.is_file() or not chunks_path.is_file():
        raise FileNotFoundError(
            f"Candidate documents.jsonl and chunks.jsonl are required in {index_dir}."
        )

    source_state = _source_state(index_dir)
    documents = _read_documents(documents_path)
    database_path = index_dir / SEARCH_DATABASE_NAME
    temporary_path = index_dir / f".{SEARCH_DATABASE_NAME}.tmp"
    if temporary_path.exists():
        temporary_path.unlink()

    connection = sqlite3.connect(temporary_path)
    chunks_indexed = 0
    malformed_rows = 0
    try:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("PRAGMA cache_size = -65536")
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE chunks (row_id INTEGER PRIMARY KEY, payload TEXT NOT NULL, searchable TEXT NOT NULL)"
        )
        connection.execute(
            """
            CREATE VIRTUAL TABLE chunk_fts USING fts5(
                searchable,
                content='chunks',
                content_rowid='row_id',
                tokenize='trigram'
            )
            """
        )

        insert_chunk = "INSERT INTO chunks(row_id, payload, searchable) VALUES (?, ?, ?)"
        insert_fts = "INSERT INTO chunk_fts(rowid, searchable) VALUES (?, ?)"
        with chunks_path.open("r", encoding="utf-8") as stream:
            for row_id, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    malformed_rows += 1
                    continue

                metadata = " ".join((
                    str(chunk.get("structure_reference") or ""),
                    str(chunk.get("parent_reference") or ""),
                    " ".join(str(value) for value in chunk.get("hierarchy_path") or []),
                    documents.get(chunk.get("doc_id"), ""),
                ))
                searchable = normalize_search_text(
                    f"{chunk.get('text', '')} {metadata}"
                )
                payload = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
                connection.execute(insert_chunk, (row_id, payload, searchable))
                connection.execute(insert_fts, (row_id, searchable))
                chunks_indexed += 1
                if chunks_indexed % 1000 == 0:
                    connection.commit()

        if _source_state(index_dir) != source_state:
            raise RuntimeError("Candidate JSONL sources changed while building the search index.")

        connection.commit()
        metadata_rows = {
            "schema_version": SEARCH_SCHEMA_VERSION,
            "source_state": json.dumps(source_state, sort_keys=True),
            "chunks_indexed": str(chunks_indexed),
            "malformed_rows": str(malformed_rows),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            metadata_rows.items(),
        )
        connection.commit()
    except Exception:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    os.replace(temporary_path, database_path)
    return {
        "database_path": str(database_path),
        "chunks_indexed": chunks_indexed,
        "malformed_rows": malformed_rows,
        "database_size_bytes": database_path.stat().st_size,
        "source_state": source_state,
    }
