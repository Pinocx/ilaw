"""Tests for append-only audit persistence and agent integration."""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from normative_repository.agent import LegalResearchAgent, SqliteAuditLedger


class AgentAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).parent / "fixtures" / "agent-audit-test.sqlite3"
        self.path.unlink(missing_ok=True)
        self.addCleanup(self.path.unlink, missing_ok=True)

    def test_hash_chain_detects_event_integrity_and_prevents_updates(self) -> None:
        ledger = SqliteAuditLedger(self.path)
        ledger.append(
            event_type="research_completed", run_id="run-1", organization_id="org-1",
            payload={"query": "question", "source_ids": ["doc::chunk"]},
        )
        ledger.append(
            event_type="answer_verified", run_id="run-1", organization_id="org-1",
            payload={"prompt_version": "v1", "token_usage": {"total_tokens": 12}},
        )
        self.assertTrue(ledger.verify_integrity()["valid"])
        self.assertEqual(len(ledger.records(organization_id="org-1")), 2)
        connection = sqlite3.connect(self.path)
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE agent_audit SET payload_json = '{}' WHERE event_id = 1")
        finally:
            connection.close()

    def test_agent_persists_query_and_source_traceability_when_audit_is_enabled(self) -> None:
        ledger = SqliteAuditLedger(self.path)
        agent = LegalResearchAgent(
            audit_ledger=ledger,
            retrieve=lambda *args, **kwargs: {
                "sources": [{
                    "source_id": "doc::chunk", "document_id": "doc", "chunk_id": "chunk",
                    "filename": "law.md", "source_pages": [1],
                }],
                "insufficient_evidence": False, "limitations": [],
            },
            verify=lambda **kwargs: {"identity_verified": True},
        )
        run = agent.run("Find Article 1.", organization_id="org-a", user_id="user-a")
        record = ledger.records(organization_id="org-a")[0]
        self.assertEqual(run.status, "evidence_retrieved")
        self.assertEqual(record["payload"]["query"], "Find Article 1.")
        self.assertEqual(record["payload"]["verified_source_ids"], ["doc::chunk"])
        self.assertTrue(ledger.verify_integrity()["valid"])


if __name__ == "__main__":
    unittest.main()
