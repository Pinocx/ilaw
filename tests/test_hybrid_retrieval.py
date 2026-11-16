"""Tests for weighted hybrid ranking and optional pgvector adapters."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from normative_repository.rag.hybrid import HybridRetriever, OllamaEmbedder, PostgresPgVectorRetriever, PostgresStairRetriever


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.statements = []

    def execute(self, query, params=None):
        self.statements.append((query, params))

    def fetchall(self):
        return self.rows

    def close(self):
        pass


class FakeConnection:
    def __init__(self, cursor):
        self.fake_cursor = cursor

    def cursor(self):
        return self.fake_cursor

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


class HybridRetrievalTests(unittest.TestCase):
    def test_stair_schema_and_index_store_hierarchy_and_source_payload(self) -> None:
        cursor = FakeCursor()
        adapter = PostgresStairRetriever(lambda: FakeConnection(cursor))
        adapter.ensure_schema()
        count = adapter.index_chunks([
            {"doc_id": "d1", "chunk_id": "c1", "text": "Artículo 5", "hierarchy_path": ["Título I", "Artículo 5"]},
        ])
        self.assertEqual(count, 1)
        sql = " ".join(statement for statement, _ in cursor.statements)
        self.assertIn("TSVECTOR", sql)
        self.assertIn("USING GIN (hierarchy_path)", sql)
        self.assertIn("gin_trgm_ops", sql)
        self.assertIn("to_tsvector('spanish'", sql)
        self.assertEqual(cursor.statements[-1][1][4], ["Título I", "Artículo 5"])

    def test_stair_index_preserves_nul_source_text_outside_postgres_text(self) -> None:
        cursor = FakeCursor()
        source = "before\x00after"
        PostgresStairRetriever(lambda: FakeConnection(cursor)).index_chunks([
            {"doc_id": "d", "chunk_id": "c", "text": source},
        ])
        params = cursor.statements[0][1]
        self.assertEqual(params[3], source.encode("utf-8"))
        self.assertNotIn("text", json.loads(params[2]))
        self.assertNotIn("\x00", params[6])
        self.assertEqual(params[5], "")

    def test_stair_search_returns_traceable_ranked_hits(self) -> None:
        cursor = FakeCursor(rows=[({"doc_id": "d", "chunk_id": "c"}, b"Art\x00culo 5", 0.5, 1.0)])
        hits = PostgresStairRetriever(lambda: FakeConnection(cursor)).search("prision preventiva", limit=5)
        self.assertEqual(hits[0]["doc_id"], "d")
        self.assertEqual(hits[0]["stair_backend"], "postgresql_structure_aware")
        self.assertEqual(hits[0]["text"], "Art\x00culo 5")
        self.assertIn("plainto_tsquery('spanish'", cursor.statements[0][0])

    def test_article_lookup_uses_indexed_hierarchy_reference(self) -> None:
        cursor = FakeCursor(rows=[({"doc_id": "d", "chunk_id": "c"}, b"source", 0.0, 1.0)])
        hits = PostgresStairRetriever(lambda: FakeConnection(cursor)).search("Artículo 5", limit=5)
        self.assertEqual(hits[0]["chunk_id"], "c")
        sql = cursor.statements[0][0]
        self.assertIn("hierarchy_text ~*", sql)
        self.assertNotIn("plainto_tsquery", sql)
        self.assertIn("5", cursor.statements[0][1][0])

    def test_explicit_article_lookup_skips_unbounded_local_fts_scan(self) -> None:
        hit = {"doc_id": "d", "chunk_id": "c", "text": "Artículo 5", "score": 1.0}
        retriever = HybridRetriever(
            index_dir=Path("unused"),
            hierarchy_retriever=type("Stair", (), {"search": lambda *_args, **_kwargs: [hit]})(),
        )
        with patch("normative_repository.rag.hybrid.search_chunks", side_effect=AssertionError("must use indexed reference lookup")):
            result = retriever.retrieve("Resume el contenido del Artículo 5 con citas verificables.", limit=5)
        self.assertEqual(result["results"][0]["chunk_id"], "c")
        self.assertEqual(result["retrieval_methods"], ["hierarchy"])

    def test_weighted_rrf_merges_rankings_and_preserves_methods(self) -> None:
        lexical_hit = {"doc_id": "d1", "chunk_id": "c1", "text": "lexical", "score": 100}
        semantic_hit = {"doc_id": "d2", "chunk_id": "c2", "text": "semantic", "score": 0.95}
        retriever = HybridRetriever(
            index_dir=Path("unused"),
            semantic_retriever=type("Semantic", (), {"search": lambda *_args, **_kwargs: [semantic_hit, lexical_hit]})(),
            lexical_weight=0.2, semantic_weight=0.8, hierarchy_weight=0, rrf_constant=1,
        )
        with patch("normative_repository.rag.hybrid.search_chunks", return_value={
            "results": [lexical_hit], "search_backend": "sqlite_fts5_trigram", "filters": {},
        }):
            result = retriever.retrieve("query", limit=5)
        self.assertEqual(result["results"][0]["doc_id"], "d2")
        self.assertEqual(result["results"][1]["retrieval_methods"], ["lexical", "semantic"])
        self.assertEqual(result["search_backend"], "hybrid_rrf")

    def test_pgvector_query_uses_bound_parameters_and_returns_traceable_payload(self) -> None:
        cursor = FakeCursor(rows=[({"doc_id": "d", "chunk_id": "c", "text": "source"}, 0.91)])
        adapter = PostgresPgVectorRetriever(lambda: FakeConnection(cursor), FakeEmbedder(), dimension=8)
        hits = adapter.search("legal question", limit=4, classification="legislation")
        self.assertEqual(hits[0]["chunk_id"], "c")
        self.assertEqual(hits[0]["semantic_score"], 0.91)
        sql, params = cursor.statements[0]
        self.assertIn("embedding <=> %s::vector", sql)
        self.assertEqual(params[1], "legislation")

    def test_pgvector_adapter_rejects_unsafe_table_and_invalid_vector_shape(self) -> None:
        with self.assertRaises(ValueError):
            PostgresPgVectorRetriever(lambda: None, FakeEmbedder(), dimension=8, table_name="vectors; DROP TABLE x")
        adapter = PostgresPgVectorRetriever(lambda: None, type("BadEmbedder", (), {"embed": lambda self, _texts: [[0.1] * 7]})(), dimension=8)
        with self.assertRaises(ValueError):
            adapter.search("query", limit=1)

    def test_pgvector_indexer_upserts_bounded_batches(self) -> None:
        connections = []

        def connect():
            cursor = FakeCursor()
            connections.append(cursor)
            return FakeConnection(cursor)

        adapter = PostgresPgVectorRetriever(connect, FakeEmbedder(), dimension=8)
        count = adapter.index_chunks([
            {"doc_id": "d1", "chunk_id": "c1", "text": "one"},
            {"doc_id": "d1", "chunk_id": "c2", "text": "two"},
            {"doc_id": "d2", "chunk_id": "c1", "text": "three"},
        ], batch_size=2)
        self.assertEqual(count, 3)
        self.assertEqual([len(cursor.statements) for cursor in connections], [2, 1])
        self.assertIn("ON CONFLICT (doc_id, chunk_id) DO UPDATE", connections[0].statements[0][0])

    def test_ollama_embedder_normalizes_batch_response(self) -> None:
        payload = {"embeddings": [[0.1] * 8, [0.2] * 8]}
        with patch("normative_repository.rag.hybrid.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
            vectors = OllamaEmbedder("http://localhost:11434", "embedding-model").embed(["one", "two"])
        self.assertEqual(len(vectors), 2)
        request = urlopen.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/api/embed"))


if __name__ == "__main__":
    unittest.main()
