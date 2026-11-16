"""Optional hybrid retrieval adapters and reciprocal-rank fusion."""

from __future__ import annotations

import json
import math
import re
import socket
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..search import search_chunks


ARTICLE_LOOKUP_RE = re.compile(
    r"\b(?:art[ií]culo|art\.?)\s*(?:n(?:[.°ºo]|ro)?\s*)?"
    r"(?P<number>\d+(?:[.-]\d+)*(?:[A-Z])?)\b",
    re.IGNORECASE,
)


class SemanticRetriever(Protocol):
    """Semantic source retriever returning source-traceable chunk records."""

    def search(self, query: str, *, limit: int, **filters: Any) -> list[dict[str, Any]]: ...


class HierarchyRetriever(Protocol):
    """STAIR adapter contract returning source-traceable hierarchical units."""

    def search(self, query: str, *, limit: int, **filters: Any) -> list[dict[str, Any]]: ...


class TextEmbedder(Protocol):
    """Batch text embedding interface consumed by the pgvector indexer."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _validate_vector(vector: Sequence[float], dimension: int) -> list[float]:
    if len(vector) != dimension or any(not math.isfinite(float(value)) for value in vector):
        raise ValueError("Embedding vector has an invalid dimension or non-finite values.")
    return [float(value) for value in vector]


class OllamaEmbedder:
    """Local Ollama `/api/embed` adapter; requests occur only when called."""

    def __init__(self, endpoint: str, model: str, *, timeout_seconds: float = 120.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        request = Request(
            f"{self.endpoint}/api/embed",
            data=json.dumps({"model": self.model, "input": list(texts)}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Embedding provider returned HTTP {exc.code}.") from None
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError("Embedding provider request timed out.") from exc
        except URLError as exc:
            raise RuntimeError("Embedding provider could not be reached.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Embedding provider returned invalid JSON.") from exc
        vectors = raw.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("Embedding provider returned an unexpected vector batch.")
        return vectors


class PostgresStairRetriever:
    """PostgreSQL-backed structure-aware legal retrieval compatible with STAIR."""

    def __init__(self, connection_factory: Callable[[], Any], *, table_name: str = "legal_hierarchy_chunks") -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", table_name):
            raise ValueError("Table name must be a simple SQL identifier.")
        self.connection_factory = connection_factory
        self.table_name = table_name

    def ensure_schema(self) -> None:
        connection = self.connection_factory()
        cursor = connection.cursor()
        try:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    source_text BYTEA,
                    hierarchy_path TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                    hierarchy_text TEXT NOT NULL DEFAULT '',
                    search_vector TSVECTOR NOT NULL,
                    PRIMARY KEY (doc_id, chunk_id)
                )
            """)
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {self.table_name}_search_gin ON {self.table_name} USING GIN (search_vector)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {self.table_name}_hierarchy_gin ON {self.table_name} USING GIN (hierarchy_path)")
            cursor.execute(f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS source_text BYTEA")
            cursor.execute(f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS hierarchy_text TEXT NOT NULL DEFAULT ''")
            cursor.execute(f"""
                UPDATE {self.table_name}
                SET hierarchy_text = concat_ws(' ', payload->>'structure_reference',
                    payload->>'parent_reference', array_to_string(hierarchy_path, ' '))
                WHERE hierarchy_text = '' AND
                    (payload ? 'structure_reference' OR payload ? 'parent_reference' OR cardinality(hierarchy_path) > 0)
            """)
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {self.table_name}_hierarchy_text_gin ON {self.table_name} USING GIN (hierarchy_text gin_trgm_ops)")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def index_chunks(self, chunks: Iterable[dict[str, Any]], *, batch_size: int = 500) -> int:
        """Upsert source-traceable chunks and their Spanish full-text vectors."""
        if not 1 <= batch_size <= 5000:
            raise ValueError("batch_size must be between 1 and 5000.")
        count = 0
        connection = self.connection_factory()
        cursor = connection.cursor()
        try:
            batch = []
            def flush() -> None:
                nonlocal count, batch
                for item in batch:
                    doc_id, chunk_id = item.get("doc_id"), item.get("chunk_id")
                    if not doc_id or not chunk_id:
                        raise ValueError("Every indexed chunk must have document and chunk IDs.")
                    path = item.get("hierarchy_path") or []
                    hierarchy_text = " ".join(str(value) for value in [
                        item.get("structure_reference") or "", item.get("parent_reference") or "", *path,
                    ] if value).replace("\x00", "")
                    source_text = str(item.get("text") or "")
                    payload = {key: value for key, value in item.items() if key != "text"}
                    cursor.execute(f"""
                        INSERT INTO {self.table_name}
                            (doc_id, chunk_id, payload, source_text, hierarchy_path, hierarchy_text, search_vector)
                        VALUES (%s, %s, %s::jsonb, %s, %s, %s, to_tsvector('spanish', %s))
                        ON CONFLICT (doc_id, chunk_id) DO UPDATE SET
                            payload = EXCLUDED.payload,
                            source_text = EXCLUDED.source_text,
                            hierarchy_path = EXCLUDED.hierarchy_path,
                            hierarchy_text = EXCLUDED.hierarchy_text,
                            search_vector = EXCLUDED.search_vector
                    """, (doc_id, chunk_id, json.dumps(payload, ensure_ascii=False),
                          source_text.encode("utf-8"), path, hierarchy_text, source_text.replace("\x00", "")))
                connection.commit()
                count += len(batch)
                batch = []
            for item in chunks:
                batch.append(item)
                if len(batch) >= batch_size:
                    flush()
            flush()
            return count
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def search(self, query: str, *, limit: int, **filters: Any) -> list[dict[str, Any]]:
        """Rank Spanish FTS matches and explicit hierarchy/reference matches."""
        if not 1 <= limit <= 100:
            raise ValueError("Hierarchy retrieval limit must be between 1 and 100.")
        connection = self.connection_factory()
        cursor = connection.cursor()
        try:
            article_match = ARTICLE_LOOKUP_RE.search(query)
            if article_match:
                number = article_match.group("number")
                reference_pattern = (
                    r"(^|[^[:alnum:]])(?:art[ií]culo|art[.]?)[[:space:]]*"
                    r"(?:n[.]?[[:space:]]*)?" + re.escape(number) + r"([^[:alnum:]]|$)"
                )
                cursor.execute(f"""
                    SELECT payload, source_text, 0.0 AS lexical_score, 1.0 AS hierarchy_score
                    FROM {self.table_name}
                    WHERE (payload->>'structure_reference' ~* %s OR hierarchy_text ~* %s)
                      AND (%s::text IS NULL OR payload->>'reviewed_classification' = %s::text)
                      AND (%s::text IS NULL OR payload->>'structure_type' = %s::text)
                      AND (%s = FALSE OR payload->>'classification_status' = 'reviewed')
                    ORDER BY (payload->>'structure_reference' ~* %s) DESC,
                             cardinality(hierarchy_path) DESC, doc_id, chunk_id
                    LIMIT %s
                """, (reference_pattern, reference_pattern,
                      filters.get("classification"), filters.get("classification"),
                      filters.get("structure_type"), filters.get("structure_type"),
                      bool(filters.get("reviewed_only", False)), reference_pattern, limit))
                results = []
                for payload, source_text, lexical_score, hierarchy_score in cursor.fetchall():
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    text = source_text.decode("utf-8") if isinstance(source_text, bytes) else source_text
                    results.append({**payload, "text": text or "", "score": 1.0,
                                    "hierarchy_score": float(hierarchy_score), "stair_backend": "postgresql_structure_aware"})
                return results
            cursor.execute(f"""
                SELECT payload, source_text,
                       ts_rank_cd(search_vector, plainto_tsquery('spanish', %s)) AS lexical_score,
                       CASE WHEN hierarchy_text ILIKE %s THEN 1.0 ELSE 0.0 END AS hierarchy_score
                FROM {self.table_name}
                WHERE (search_vector @@ plainto_tsquery('spanish', %s)
                       OR hierarchy_text ILIKE %s)
                  AND (%s::text IS NULL OR payload->>'reviewed_classification' = %s::text)
                  AND (%s::text IS NULL OR payload->>'structure_type' = %s::text)
                  AND (%s = FALSE OR payload->>'classification_status' = 'reviewed')
                ORDER BY hierarchy_score DESC, lexical_score DESC, doc_id, chunk_id
                LIMIT %s
            """, (query, f"%{query}%", query, f"%{query}%", filters.get("classification"),
                  filters.get("classification"), filters.get("structure_type"), filters.get("structure_type"),
                  bool(filters.get("reviewed_only", False)), limit))
            results = []
            for payload, source_text, lexical_score, hierarchy_score in cursor.fetchall():
                if isinstance(payload, str):
                    payload = json.loads(payload)
                text = source_text.decode("utf-8") if isinstance(source_text, bytes) else source_text
                results.append({**payload, "text": text or "", "score": float(lexical_score) + float(hierarchy_score),
                                "hierarchy_score": float(hierarchy_score), "stair_backend": "postgresql_structure_aware"})
            return results
        finally:
            cursor.close()
            connection.close()

class PostgresPgVectorRetriever:
    """PostgreSQL pgvector index for derived chunk embeddings and traceable payloads."""

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        embedder: TextEmbedder,
        *,
        dimension: int,
        table_name: str = "legal_chunk_vectors",
    ) -> None:
        if not 8 <= dimension <= 2000:
            raise ValueError("Vector dimension must be between 8 and 2000 for pgvector HNSW indexing.")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", table_name):
            raise ValueError("Table name must be a simple SQL identifier.")
        self.connection_factory = connection_factory
        self.embedder = embedder
        self.dimension = dimension
        self.table_name = table_name

    def ensure_schema(self) -> None:
        connection = self.connection_factory()
        cursor = connection.cursor()
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    doc_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
                    embedding VECTOR({self.dimension}) NOT NULL, payload JSONB NOT NULL,
                    classification TEXT, structure_type TEXT, reviewed BOOLEAN NOT NULL DEFAULT FALSE,
                    PRIMARY KEY (doc_id, chunk_id)
                )
            """)
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw ON {self.table_name} USING hnsw (embedding vector_cosine_ops)")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def index_chunks(self, chunks: Iterable[dict[str, Any]], *, batch_size: int = 64) -> int:
        """Upsert derived vectors in bounded batches; do not alter source artifacts."""
        if not 1 <= batch_size <= 512:
            raise ValueError("batch_size must be between 1 and 512.")
        count = 0
        batch: list[dict[str, Any]] = []

        def write_batch(records: list[dict[str, Any]]) -> int:
            if not records:
                return 0
            vectors = self.embedder.embed([str(item.get("text") or "") for item in records])
            if len(vectors) != len(records):
                raise ValueError("Embedder returned a different number of vectors than input chunks.")
            connection = self.connection_factory()
            cursor = connection.cursor()
            try:
                for item, vector in zip(records, vectors):
                    doc_id, chunk_id = item.get("doc_id"), item.get("chunk_id")
                    if not doc_id or not chunk_id:
                        raise ValueError("Every indexed chunk must have document and chunk IDs.")
                    normalized = _validate_vector(vector, self.dimension)
                    cursor.execute(f"""
                        INSERT INTO {self.table_name}
                            (doc_id, chunk_id, embedding, payload, classification, structure_type, reviewed)
                        VALUES (%s, %s, %s::vector, %s::jsonb, %s, %s, %s)
                        ON CONFLICT (doc_id, chunk_id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            payload = EXCLUDED.payload,
                            classification = EXCLUDED.classification,
                            structure_type = EXCLUDED.structure_type,
                            reviewed = EXCLUDED.reviewed
                    """, (
                        doc_id, chunk_id, json.dumps(normalized), json.dumps(item, ensure_ascii=False),
                        item.get("reviewed_classification") or item.get("suggested_classification"),
                        item.get("structure_type"), item.get("classification_status") == "reviewed",
                    ))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                cursor.close()
                connection.close()
            return len(records)

        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= batch_size:
                count += write_batch(batch)
                batch = []
        count += write_batch(batch)
        return count

    def search(self, query: str, *, limit: int, **filters: Any) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise ValueError("Semantic retrieval limit must be between 1 and 100.")
        vector_rows = self.embedder.embed([query])
        if len(vector_rows) != 1:
            raise ValueError("Embedder must return one vector for one query.")
        vector = json.dumps(_validate_vector(vector_rows[0], self.dimension))
        classification = filters.get("classification")
        structure_type = filters.get("structure_type")
        reviewed_only = bool(filters.get("reviewed_only", False))
        connection = self.connection_factory()
        cursor = connection.cursor()
        try:
            cursor.execute(f"""
                SELECT payload, 1 - (embedding <=> %s::vector) AS similarity
                FROM {self.table_name}
                WHERE (%s IS NULL OR classification = %s)
                  AND (%s IS NULL OR structure_type = %s)
                  AND (%s = FALSE OR reviewed = TRUE)
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (vector, classification, classification, structure_type, structure_type, reviewed_only, vector, limit))
            rows = cursor.fetchall()
            hits = []
            for payload, similarity in rows:
                if isinstance(payload, str):
                    payload = json.loads(payload)
                hits.append({**payload, "score": float(similarity), "semantic_score": float(similarity)})
            return hits
        finally:
            cursor.close()
            connection.close()


class HybridRetriever:
    """Fuse lexical, semantic, and optional STAIR rankings with weighted RRF."""

    def __init__(
        self,
        *,
        index_dir,
        semantic_retriever: SemanticRetriever | None = None,
        hierarchy_retriever: HierarchyRetriever | None = None,
        lexical_weight: float = 0.5,
        semantic_weight: float = 0.35,
        hierarchy_weight: float = 0.15,
        rrf_constant: int = 60,
    ) -> None:
        weights = (lexical_weight, semantic_weight, hierarchy_weight)
        if any(weight < 0 or weight > 1 for weight in weights) or not any(weights):
            raise ValueError("Retrieval weights must be within [0, 1] and not all zero.")
        if rrf_constant < 1:
            raise ValueError("rrf_constant must be positive.")
        self.index_dir = index_dir
        self.semantic_retriever = semantic_retriever
        self.hierarchy_retriever = hierarchy_retriever
        self.weights = weights
        self.rrf_constant = rrf_constant

    def retrieve(self, query: str, *, limit: int, **filters: Any) -> dict[str, Any]:
        if not 1 <= limit <= 100:
            raise ValueError("Hybrid retrieval limit must be between 1 and 100.")
        exact_article_lookup = bool(self.hierarchy_retriever and ARTICLE_LOOKUP_RE.search(query))
        lexical = (
            {"results": [], "search_backend": "postgresql_reference_lookup", "filters": filters, "total_matches": 0}
            if exact_article_lookup
            else search_chunks(query, index_dir=self.index_dir, limit=limit, **filters)
        )
        ranked_lists: list[tuple[str, float, list[dict[str, Any]]]] = []
        if not exact_article_lookup:
            ranked_lists.append(("lexical", self.weights[0], lexical.get("results", [])))
        if self.semantic_retriever is not None:
            ranked_lists.append(("semantic", self.weights[1], self.semantic_retriever.search(query, limit=limit, **filters)))
        if self.hierarchy_retriever is not None:
            ranked_lists.append(("hierarchy", self.weights[2], self.hierarchy_retriever.search(query, limit=limit, **filters)))
        fused: dict[tuple[str, str], dict[str, Any]] = {}
        for method, weight, hits in ranked_lists:
            for rank, hit in enumerate(hits, start=1):
                key = (str(hit.get("doc_id") or ""), str(hit.get("chunk_id") or ""))
                if not all(key):
                    continue
                record = fused.setdefault(key, {**hit, "rrf_score": 0.0, "retrieval_methods": []})
                record["rrf_score"] += weight / (self.rrf_constant + rank)
                if method not in record["retrieval_methods"]:
                    record["retrieval_methods"].append(method)
                if method == "lexical":
                    record.update(hit)
        results = sorted(fused.values(), key=lambda item: (-item["rrf_score"], item["doc_id"], item["chunk_id"]))[:limit]
        for hit in results:
            hit["score"] = hit["rrf_score"]
        return {
            **lexical,
            "results": results,
            "total_matches": len(fused),
            "search_backend": "hybrid_rrf" if len(ranked_lists) > 1 else lexical.get("search_backend", "unknown"),
            "retrieval_methods": [method for method, _, _ in ranked_lists],
            "ranking_weights": {method: weight for method, weight, _ in ranked_lists},
        }
